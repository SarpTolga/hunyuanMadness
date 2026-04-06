import os
import sys
import time
import uuid
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename

# Add Hunyuan3D repo to path
HUNYUAN_PATH = "D:/Hunyuan3D"
sys.path.insert(0, HUNYUAN_PATH)
os.chdir(HUNYUAN_PATH)

app = Flask(__name__, static_folder="static")

UPLOAD_DIR = Path("D:/Hunyuan3DStudio/uploads")
OUTPUT_DIR = Path("D:/Hunyuan3DStudio/outputs")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

QUALITY_PRESETS = {
    "low": {"octree_resolution": 196, "steps": 20, "guidance_scale": 5.0, "num_chunks": 8000},
    "medium": {"octree_resolution": 256, "steps": 30, "guidance_scale": 7.5, "num_chunks": 200000},
    "high": {"octree_resolution": 384, "steps": 50, "guidance_scale": 7.5, "num_chunks": 200000},
}

# Global model references (loaded once at startup)
shape_pipeline = None
paint_pipeline = None
rembg_worker = None


def load_models():
    """Load AI models into GPU memory."""
    global shape_pipeline, paint_pipeline, rembg_worker
    import torch

    print("[*] Loading background remover...")
    from hy3dgen.rembg import BackgroundRemover
    rembg_worker = BackgroundRemover()

    print("[*] Loading shape generation model (Mini)...")
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini",
        use_safetensors=True,
    )

    # Try loading texture model (will fail gracefully on low VRAM)
    try:
        print("[*] Loading texture model...")
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
        paint_pipeline.enable_model_cpu_offload()
        print("[+] Texture model loaded!")
    except Exception as e:
        print(f"[!] Texture model not available: {e}")
        print("[!] Shape-only mode active. Texture needs 16GB+ VRAM or cloud GPU.")
        paint_pipeline = None

    torch.cuda.empty_cache()
    print("[+] Models ready!")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Routes ---

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


@app.route("/api/status", methods=["GET"])
def status():
    """Return what capabilities are available."""
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    return jsonify({
        "shape_ready": shape_pipeline is not None,
        "texture_ready": paint_pipeline is not None,
        "gpu": gpu_name,
        "vram_gb": round(gpu_mem, 1),
    })


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate 3D model from uploaded image."""
    import torch
    from PIL import Image

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, or WebP."}), 400

    quality = request.form.get("quality", "medium")
    export_format = request.form.get("format", "glb")
    enable_texture = request.form.get("texture", "false") == "true"

    if quality not in QUALITY_PRESETS:
        return jsonify({"error": f"Invalid quality: {quality}"}), 400

    preset = QUALITY_PRESETS[quality]
    job_id = str(uuid.uuid4())[:8]
    print(f"\n[{job_id}] New job: quality={quality}, format={export_format}, texture={enable_texture}")

    # Save uploaded image
    filename = f"{job_id}_{secure_filename(file.filename)}"
    img_path = UPLOAD_DIR / filename
    file.save(str(img_path))

    try:
        # Load and prep image
        image = Image.open(str(img_path)).convert("RGBA")
        if image.mode == "RGB" or True:
            print(f"[{job_id}] Removing background...")
            image = rembg_worker(image.convert("RGB"))

        # Generate shape
        print(f"[{job_id}] Generating shape ({quality})...")
        t0 = time.time()
        generator = torch.Generator().manual_seed(42)
        mesh = shape_pipeline(
            image=image,
            num_inference_steps=preset["steps"],
            guidance_scale=preset["guidance_scale"],
            octree_resolution=preset["octree_resolution"],
            num_chunks=preset["num_chunks"],
            generator=generator,
        )[0]
        shape_time = round(time.time() - t0, 1)
        print(f"[{job_id}] Shape done in {shape_time}s")

        texture_time = None

        # Try texture if requested and available
        if enable_texture and paint_pipeline is not None:
            try:
                print(f"[{job_id}] Generating texture...")
                t1 = time.time()
                mesh = paint_pipeline(mesh, image=image)
                texture_time = round(time.time() - t1, 1)
                print(f"[{job_id}] Texture done in {texture_time}s")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    return jsonify({
                        "error": "Out of VRAM for texture generation. Your GPU needs 16GB+. Shape was generated but texture requires a cloud GPU.",
                        "vram_error": True,
                    }), 507
                raise
        elif enable_texture and paint_pipeline is None:
            return jsonify({
                "error": "Texture model not loaded. Needs 16GB+ VRAM. Try shape-only or use a cloud GPU.",
                "vram_error": True,
            }), 507

        # Export
        out_name = f"{job_id}.{export_format}"
        out_path = OUTPUT_DIR / out_name

        if export_format == "fbx":
            # FBX via pymeshlab
            try:
                import pymeshlab
                temp_obj = OUTPUT_DIR / f"{job_id}_temp.obj"
                mesh.export(str(temp_obj))
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(str(temp_obj))
                ms.save_current_mesh(str(out_path))
                temp_obj.unlink(missing_ok=True)
                # Clean up mtl/texture files from obj export
                for f in OUTPUT_DIR.glob(f"{job_id}_temp*"):
                    f.unlink(missing_ok=True)
            except Exception as e:
                return jsonify({"error": f"FBX export failed: {e}. Try GLB or OBJ instead."}), 500
        else:
            mesh.export(str(out_path))

        torch.cuda.empty_cache()

        stats = {
            "job_id": job_id,
            "file": out_name,
            "format": export_format,
            "quality": quality,
            "shape_time": shape_time,
            "texture_time": texture_time,
            "faces": int(mesh.faces.shape[0]) if hasattr(mesh, "faces") else 0,
            "vertices": int(mesh.vertices.shape[0]) if hasattr(mesh, "vertices") else 0,
        }
        print(f"[{job_id}] Done! {stats['faces']} faces, {stats['vertices']} verts")
        return jsonify(stats)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return jsonify({"error": "GPU out of memory. Try 'Low' quality or restart the app.", "vram_error": True}), 507
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/download/<filename>")
def download(filename):
    """Download a generated model file."""
    path = OUTPUT_DIR / secure_filename(filename)
    if not path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(path), as_attachment=True)


@app.route("/api/preview/<filename>")
def preview(filename):
    """Serve model file for 3D preview."""
    path = OUTPUT_DIR / secure_filename(filename)
    if not path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(path))


if __name__ == "__main__":
    print("=" * 50)
    print("  Hunyuan3D Studio")
    print("=" * 50)
    load_models()
    print(f"\n  Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
