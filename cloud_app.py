import os,sys,time,uuid,traceback,threading,json,multiprocessing,gc
from pathlib import Path
from flask import Flask,request,jsonify,send_from_directory,send_file
from werkzeug.utils import secure_filename

app=Flask(__name__,static_folder="static")
UPLOAD_DIR=Path("uploads")
OUTPUT_DIR=Path("outputs")
JOBS_DIR=Path("jobs")
ALLOWED_EXTENSIONS={"png","jpg","jpeg","webp"}

# Presets optimized for Unity low-poly game models
QUALITY_PRESETS={
    "fast":{
        "steps":5,
        "octree_resolution":196,
        "guidance_scale":5.0,
        "num_chunks":20000,
        "max_faces":5000,
        "texture_size":512,
        "render_size":512,
    },
    "balanced":{
        "steps":15,
        "octree_resolution":256,
        "guidance_scale":7.0,
        "num_chunks":20000,
        "max_faces":10000,
        "texture_size":1024,
        "render_size":1024,
    },
    "detailed":{
        "steps":30,
        "octree_resolution":384,
        "guidance_scale":7.5,
        "num_chunks":20000,
        "max_faces":30000,
        "texture_size":1024,
        "render_size":1024,
    },
}

shape_pipeline=None
paint_pipeline=None
rembg_worker=None
face_reducer=None
floater_remover=None
degen_remover=None
gen_lock=threading.Lock()

def write_job(job_id,data):
    p=JOBS_DIR/f"{job_id}.json"
    p.write_text(json.dumps(data))

def read_job(job_id):
    p=JOBS_DIR/f"{job_id}.json"
    if not p.exists():return None
    try:return json.loads(p.read_text())
    except:return None

def update_job(job_id,**kwargs):
    data=read_job(job_id) or {}
    data.update(kwargs)
    write_job(job_id,data)

def load_models():
    global shape_pipeline,paint_pipeline,rembg_worker,face_reducer,floater_remover,degen_remover
    import torch

    print("[*] Loading background remover...")
    from hy3dgen.rembg import BackgroundRemover
    rembg_worker=BackgroundRemover()

    print("[*] Loading shape model (Mini Turbo)...")
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    shape_pipeline=Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini",
        use_safetensors=True,
    )

    # Enable FlashVDM for much faster mesh extraction
    try:
        shape_pipeline.enable_flashvdm(topk_mode='mean')
        print("[+] FlashVDM enabled (faster mesh extraction)")
    except Exception as e:
        print(f"[!] FlashVDM not available: {e}")

    # Load postprocessors
    print("[*] Loading postprocessors...")
    try:
        from hy3dgen.shapegen.postprocessors import FaceReducer,FloaterRemover,DegenerateFaceRemover
        face_reducer=FaceReducer()
        floater_remover=FloaterRemover()
        degen_remover=DegenerateFaceRemover()
        print("[+] Postprocessors ready (FaceReducer, FloaterRemover, DegenerateFaceRemover)")
    except Exception as e:
        print(f"[!] Postprocessors failed: {e}")

    print("[*] Loading texture model...")
    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        paint_pipeline=Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
        # NOTE: Do NOT use enable_model_cpu_offload() on cloud GPUs with 24GB+
        # It shuffles data CPU<->GPU and makes everything 5-10x slower
        print("[+] Texture model loaded (full GPU)!")
    except Exception as e:
        print(f"[!] Texture failed: {e}")
        paint_pipeline=None

    torch.cuda.empty_cache()
    print("[+] All models ready!")

def allowed_file(f):
    return "." in f and f.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

def clear_gpu():
    """Aggressive GPU memory cleanup between runs."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def run_generation(job_id,img_path,quality,export_format,enable_texture,target_faces):
    import torch
    from PIL import Image

    # Only one generation at a time — prevents VRAM conflicts
    with gen_lock:
        preset=QUALITY_PRESETS.get(quality,QUALITY_PRESETS["balanced"])
        max_faces=target_faces or preset["max_faces"]
        try:
            clear_gpu()

            # Remove background
            update_job(job_id,status="removing_bg",heartbeat=time.time())
            image=Image.open(str(img_path)).convert("RGBA")
            print(f"[{job_id}] Removing background...")
            image=rembg_worker(image.convert("RGB"))

            # Generate shape
            update_job(job_id,status="generating_shape",heartbeat=time.time())
            print(f"[{job_id}] Generating shape ({quality}, {preset['steps']} steps, res={preset['octree_resolution']})...")
            t0=time.time()
            generator=torch.Generator().manual_seed(42)
            mesh=shape_pipeline(
                image=image,
                num_inference_steps=preset["steps"],
                guidance_scale=preset["guidance_scale"],
                octree_resolution=preset["octree_resolution"],
                num_chunks=preset["num_chunks"],
                generator=generator,
            )[0]
            torch.cuda.synchronize()
            shape_time=round(time.time()-t0,1)
            raw_faces=int(mesh.faces.shape[0]) if hasattr(mesh,"faces") else 0
            print(f"[{job_id}] Shape done in {shape_time}s ({raw_faces} raw faces)")
            update_job(job_id,shape_time=shape_time,raw_faces=raw_faces,heartbeat=time.time())

            # Post-processing: clean mesh
            update_job(job_id,status="cleaning_mesh",heartbeat=time.time())
            if floater_remover:
                print(f"[{job_id}] Removing floaters...")
                mesh=floater_remover(mesh)
            if degen_remover:
                print(f"[{job_id}] Removing degenerate faces...")
                mesh=degen_remover(mesh)

            # Face reduction BEFORE texture — UV unwrapping 600K faces is extremely slow
            if face_reducer and max_faces>0:
                cur_faces=int(mesh.faces.shape[0]) if hasattr(mesh,"faces") else 0
                if cur_faces>max_faces:
                    update_job(job_id,status="reducing_faces",heartbeat=time.time())
                    print(f"[{job_id}] Reducing faces: {cur_faces} -> {max_faces}...")
                    mesh=face_reducer(mesh,max_facenum=max_faces)
                    reduced_faces=int(mesh.faces.shape[0]) if hasattr(mesh,"faces") else 0
                    print(f"[{job_id}] Reduced to {reduced_faces} faces")

            # Texture on the reduced mesh (much faster UV unwrap + bake)
            texture_time=None
            if enable_texture and paint_pipeline:
                update_job(job_id,status="generating_texture",heartbeat=time.time())
                print(f"[{job_id}] Generating texture (render={preset['render_size']}, tex={preset['texture_size']})...")
                clear_gpu()

                # Configure texture resolution
                if hasattr(paint_pipeline,'config'):
                    paint_pipeline.config.render_size=preset["render_size"]
                    paint_pipeline.config.texture_size=preset["texture_size"]

                t1=time.time()
                mesh=paint_pipeline(mesh,image=image)
                torch.cuda.synchronize()
                texture_time=round(time.time()-t1,1)
                print(f"[{job_id}] Texture done in {texture_time}s")
                update_job(job_id,texture_time=texture_time,heartbeat=time.time())
                clear_gpu()
            elif enable_texture:
                update_job(job_id,status="error",error="Texture model not loaded")
                return

            # Export
            update_job(job_id,status="exporting",heartbeat=time.time())
            out_name=f"{job_id}.{export_format}"
            out_path=OUTPUT_DIR/out_name
            if export_format=="fbx":
                try:
                    import pymeshlab
                    temp=OUTPUT_DIR/f"{job_id}_temp.obj"
                    mesh.export(str(temp))
                    ms=pymeshlab.MeshSet()
                    ms.load_new_mesh(str(temp))
                    ms.save_current_mesh(str(out_path))
                    temp.unlink(missing_ok=True)
                    for f2 in OUTPUT_DIR.glob(f"{job_id}_temp*"):f2.unlink(missing_ok=True)
                except Exception as e:
                    update_job(job_id,status="error",error=f"FBX export failed: {e}")
                    return
            else:
                mesh.export(str(out_path))

            clear_gpu()
            faces=int(mesh.faces.shape[0]) if hasattr(mesh,"faces") else 0
            verts=int(mesh.vertices.shape[0]) if hasattr(mesh,"vertices") else 0
            print(f"[{job_id}] Done! {faces} faces, {verts} verts")
            update_job(job_id,status="done",file=out_name,faces=faces,vertices=verts,
                       raw_faces=raw_faces,max_faces=max_faces,heartbeat=time.time())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                clear_gpu()
            traceback.print_exc()
            update_job(job_id,status="error",error=str(e))
        except Exception as e:
            traceback.print_exc()
            update_job(job_id,status="error",error=str(e))

# === Flask routes ===
@app.route("/")
def index():
    return send_from_directory("static","index.html")
@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static",path)
@app.route("/api/status")
def status():
    import torch
    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    vram=torch.cuda.get_device_properties(0).total_memory/1024**3 if torch.cuda.is_available() else 0
    return jsonify({
        "shape_ready":shape_pipeline is not None,
        "texture_ready":paint_pipeline is not None,
        "gpu":gpu,
        "vram_gb":round(vram,1),
        "flashvdm":hasattr(shape_pipeline,'flashvdm_enabled') and shape_pipeline.flashvdm_enabled if shape_pipeline else False,
        "postprocessing":face_reducer is not None,
    })
@app.route("/api/generate",methods=["POST"])
def generate():
    if "image" not in request.files:
        return jsonify({"error":"No image"}),400
    file=request.files["image"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error":"Invalid file type"}),400
    quality=request.form.get("quality","balanced")
    export_format=request.form.get("format","glb")
    enable_texture=request.form.get("texture","false")=="true"
    target_faces=int(request.form.get("faces","0"))
    job_id=str(uuid.uuid4())[:8]
    print(f"\n[{job_id}] quality={quality}, format={export_format}, texture={enable_texture}, faces={target_faces}")
    filename=f"{job_id}_{secure_filename(file.filename)}"
    img_path=UPLOAD_DIR/filename
    file.save(str(img_path))
    write_job(job_id,{"status":"queued","quality":quality,"format":export_format,
                      "shape_time":None,"texture_time":None,"error":None,"heartbeat":time.time()})
    t=threading.Thread(target=run_generation,args=(job_id,img_path,quality,export_format,enable_texture,target_faces),daemon=True)
    t.start()
    return jsonify({"job_id":job_id})
@app.route("/api/job/<job_id>")
def job_status(job_id):
    data=read_job(job_id)
    if not data:return jsonify({"error":"Job not found"}),404
    hb=data.get("heartbeat",0)
    data["stale"]=data.get("status") not in ("done","error") and time.time()-hb>120
    return jsonify(data)
@app.route("/api/download/<filename>")
def download(filename):
    path=OUTPUT_DIR/secure_filename(filename)
    if not path.exists():return jsonify({"error":"Not found"}),404
    return send_file(str(path),as_attachment=True)
@app.route("/api/preview/<filename>")
def preview(filename):
    path=OUTPUT_DIR/secure_filename(filename)
    if not path.exists():return jsonify({"error":"Not found"}),404
    return send_file(str(path))

# === Proxy server on port 3333 ===
def run_proxy(proxy_port=3333,flask_port=3334):
    from http.server import HTTPServer,BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn
    from urllib.request import urlopen,Request
    jobs_dir=Path("jobs")

    class ProxyHandler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass

        def do_GET(self):
            try:
                if self.path.startswith("/api/job/"):
                    job_id=self.path.split("/")[-1]
                    p=jobs_dir/f"{job_id}.json"
                    if not p.exists():
                        self._json(404,{"error":"Job not found"})
                        return
                    data=json.loads(p.read_text())
                    hb=data.get("heartbeat",0)
                    data["stale"]=data.get("status") not in ("done","error") and time.time()-hb>120
                    self._json(200,data)
                    return
                self._proxy("GET")
            except:pass

        def do_POST(self):
            try:self._proxy("POST")
            except:pass

        def _json(self,code,data):
            body=json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _proxy(self,method):
            url=f"http://127.0.0.1:{flask_port}{self.path}"
            body=None
            if method=="POST":
                length=int(self.headers.get("Content-Length",0))
                body=self.rfile.read(length) if length else None
            req=Request(url,data=body,method=method)
            for h in ["Content-Type","Accept"]:
                v=self.headers.get(h)
                if v:req.add_header(h,v)
            try:
                resp=urlopen(req,timeout=30)
            except:
                self._json(502,{"error":"Server busy, try again"})
                return
            resp_body=resp.read()
            self.send_response(resp.status)
            for h in ["Content-Type","Content-Disposition"]:
                v=resp.headers.get(h)
                if v:self.send_header(h,v)
            self.send_header("Content-Length",str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

    class ThreadedHTTPServer(ThreadingMixIn,HTTPServer):
        daemon_threads=True

    server=ThreadedHTTPServer(("0.0.0.0",proxy_port),ProxyHandler)
    print(f"[+] Proxy server on port {proxy_port}")
    server.serve_forever()

if __name__=="__main__":
    JOBS_DIR.mkdir(exist_ok=True)
    print("="*50)
    print("  Hunyuan3D Studio (Cloud - Unity Optimized)")
    print("="*50)
    proxy=multiprocessing.Process(target=run_proxy,args=(3333,3334),daemon=True)
    proxy.start()
    load_models()
    print("\n  Server ready on port 3333\n")
    app.run(host="127.0.0.1",port=3334,debug=False,threaded=True)
