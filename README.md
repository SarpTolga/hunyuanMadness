# Hunyuan3D Studio

A web-based tool for generating 3D models from images or text using [Tencent's Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2). Optimized for Unity game development — fast, low-poly models with textures.

![Unity Optimized](https://img.shields.io/badge/Unity-Optimized-green) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Image to 3D** — Upload an image, get a textured 3D model
- **Text to 3D** — Type a description, AI generates the image and 3D model
- **Fast generation** — Shape in ~1-2s, full textured model in ~22-47s on RTX 4090
- **Face reduction** — Adjustable slider (5K-50K faces) for game-ready models
- **Multiple formats** — GLB, OBJ, STL, FBX export
- **3D preview** — Interactive preview in browser before downloading
- **Seed control** — Reproduce results or randomize for variations
- **Quality presets** — Fast (5 steps), Balanced (15 steps), Detailed (30 steps)

## Performance

Tested on RTX 4090 (24GB VRAM):

| Preset | Shape | Texture | Total |
|--------|-------|---------|-------|
| Fast + texture + 10K faces | 1.2s | ~20s | **~22s** |
| Balanced + texture + 10K faces | ~3s | ~25s | **~30s** |
| Detailed + texture + 30K faces | ~8s | ~35s | **~47s** |

Key optimizations:
- **FlashVDM** — 60-80% faster mesh extraction
- **Face reduction before texture** — UV wrapping 10K faces vs 600K (89s vs ~1s)
- **Patched texture pipeline** — Reduced diffusion steps (delight 50→15, multiview 30→12)
- **No CPU offload** on cloud GPUs — Keeps everything on GPU

## Quick Start (Cloud GPU)

Rent a GPU on [Vast.ai](https://vast.ai) (RTX 3090/4090, 80GB+ disk, PyTorch template), then:

```bash
curl -sL https://raw.githubusercontent.com/SarpTolga/hunyuanMadness/main/cloud_setup.sh | bash
```

One command. Installs everything, patches for speed, starts the server.
Then create a tunnel to `http://localhost:3333` from Vast.ai dashboard.

See [CLOUD_SETUP.md](CLOUD_SETUP.md) for the detailed step-by-step guide.

## Project Structure

```
cloud_app.py      — Cloud server (shape + texture + text-to-3D)
app.py            — Local server (shape-only, for low VRAM GPUs)
cloud_setup.sh    — One-command cloud setup script
static/index.html — Web UI
CLOUD_SETUP.md    — Detailed cloud setup guide
```

## How It Works

1. **Input** — Upload an image or type a text description
2. **Text-to-Image** (text mode only) — HunyuanDiT generates a reference image
3. **Background removal** — Removes background for clean 3D generation
4. **Shape generation** — Hunyuan3D-2 Mini with FlashVDM creates the 3D mesh
5. **Mesh cleanup** — FloaterRemover + DegenerateFaceRemover
6. **Face reduction** — Reduces to target face count (game-ready)
7. **Texture generation** — Hunyuan3D Paint pipeline bakes textures
8. **Export** — GLB/OBJ/STL/FBX download

## Requirements

**Cloud (recommended):**
- Vast.ai account (~$0.20/hr for RTX 3090)
- Everything is installed automatically by the setup script

**Local (shape-only):**
- NVIDIA GPU with 6GB+ VRAM
- Python 3.10+
- [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) installed locally

## Built With

- [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) — Tencent's open-source 3D generation model
- [HunyuanDiT](https://github.com/Tencent/HunyuanDiT) — Text-to-image model for text-to-3D
- [Flask](https://flask.palletsprojects.com/) — Web server
- [model-viewer](https://modelviewer.dev/) — 3D preview in browser

## License

This project is for non-commercial use, following the [Hunyuan3D license](https://github.com/Tencent/Hunyuan3D-2/blob/main/LICENSE).
