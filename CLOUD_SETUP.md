# Hunyuan3D Studio - Cloud GPU Setup Guide

## What This Is
Your local RTX 2060 (6GB) can do **shape-only** 3D generation (no texture/color).
For **textured 3D models**, you rent a cloud GPU with 24GB+ VRAM on Vast.ai.

---

## What You Need to Rent

| Setting | Value |
|---------|-------|
| **Platform** | [Vast.ai](https://vast.ai) |
| **GPU** | 1x RTX 3090 (24GB VRAM) |
| **Disk** | **80GB minimum** |
| **Template** | PyTorch (Vast) |
| **Cost** | ~$0.20/hr |
| **Spending limit** | Set to $5 in Account > Billing |

---

## Quick Setup (One Command)

After renting your instance and opening Jupyter Terminal, just run:

```bash
curl -s https://paste.rs/Azh69 | bash
```

This does everything automatically (clone, install, build, download app, fix Caddy, start server).
Wait for `[+] All models ready!` then create a tunnel to `http://localhost:3333`.

> **If paste.rs URLs expired:** You need to re-upload 3 files from your PC and update the URLs:
> - `D:\Hunyuan3DStudio\cloud_setup.sh` --> the setup script itself (contains URLs for the other 2 files)
> - `D:\Hunyuan3DStudio\cloud_app.py` --> the server app
> - `D:\Hunyuan3DStudio\static\index.html` --> the web UI
> Upload each to https://paste.rs, update URLs in `cloud_setup.sh`, re-upload the script, then use the new script URL.

---

## Manual Setup Steps (if you prefer step-by-step)

### Step 1: Rent the Instance
1. Go to https://vast.ai, log in
2. Go to **Search** page
3. Click **"Select Template"** > pick **"PyTorch (Vast)"**
4. Filter for 1x RTX 3090, **80GB+ disk**
5. Click **RENT** on the cheapest one
6. Wait 1-2 min for it to boot

### Step 2: Open Terminal
1. Go to **Instances** tab
2. Click **"Open"** on your instance
3. Click **"Launch Application"** on **"Jupyter Terminal"**

---

### Step 3: Clone & Install

> **Where:** Jupyter Terminal

```bash
cd /workspace
```

```bash
git clone https://github.com/Tencent/Hunyuan3D-2.git
```

```bash
cd Hunyuan3D-2
```

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

```bash
pip install flask trimesh pymeshlab pygltflib xatlas
```

```bash
apt-get update && apt-get install -y libopengl0
```

> **Why libopengl0?** pymeshlab needs OpenGL for FBX export. Without it, FBX conversion crashes.

---

### Step 4: Fix CUDA Headers

> **Where:** Jupyter Terminal

First find where cusparse.h is:
```bash
find / -name "cusparse.h" 2>/dev/null
```

It will print a path like `/venv/main/lib/python3.12/site-packages/nvidia/cu13/include/cusparse.h`.
Use the **folder path** (without the filename) in these commands:

```bash
export CPLUS_INCLUDE_PATH=/venv/main/lib/python3.12/site-packages/nvidia/cu13/include:$CPLUS_INCLUDE_PATH
```

```bash
export C_INCLUDE_PATH=/venv/main/lib/python3.12/site-packages/nvidia/cu13/include:$C_INCLUDE_PATH
```

```bash
export CUDA_HOME=/usr/local/cuda
```

> **Note:** Replace the path above with whatever `find` returned for you.

---

### Step 5: Build Texture Components

> **Where:** Jupyter Terminal

```bash
cd hy3dgen/texgen/custom_rasterizer && python setup.py install && cd ../../..
```

```bash
cd hy3dgen/texgen/differentiable_renderer && python setup.py install && cd ../../..
```

---

### Step 6: Create App Directories

> **Where:** Jupyter Terminal

```bash
mkdir -p outputs uploads static jobs
```

---

### Step 7: Download App Files

> **Where:** Jupyter Notebook (NOT Terminal)
>
> Go to Vast.ai applications > Click **"Launch Application"** on **"Jupyter"** (not Terminal) > Click **"New" > "Python 3"**
>
> Paste this in the cell and press **Shift+Enter**:

```python
import urllib.request
urllib.request.urlretrieve("https://paste.rs/vVVMK", "/workspace/Hunyuan3D-2/app.py")
urllib.request.urlretrieve("https://paste.rs/fSZg7", "/workspace/Hunyuan3D-2/static/index.html")
print("Both files downloaded!")
import os
print("app.py:", os.path.getsize("/workspace/Hunyuan3D-2/app.py"), "bytes")
print("index.html:", os.path.getsize("/workspace/Hunyuan3D-2/static/index.html"), "bytes")
```

> **If paste.rs URLs expired:** Re-upload from your PC:
> - `D:\Hunyuan3DStudio\cloud_app.py` --> upload to https://paste.rs --> use new URL for app.py
> - `D:\Hunyuan3DStudio\static\index.html` --> upload to https://paste.rs --> use new URL for index.html

---

### Step 8: Fix Caddy Proxy

> **Where:** Jupyter Terminal
>
> Vast.ai uses Caddy reverse proxy that blocks common ports. We redirect it to our app port.

```bash
sed -i 's/localhost:18384/localhost:3333/g' /etc/Caddyfile && caddy reload --config /etc/Caddyfile
```

You should see: `INFO using config from file {"file": "/etc/Caddyfile"}`

---

### Step 9: Start the Server

> **Where:** Jupyter Terminal

```bash
cd /workspace/Hunyuan3D-2 && python app.py
```

Wait for it to print:
```
[+] All models ready!

  Server ready on port 3333
```

> First run downloads ~16GB of models. Takes 5-10 minutes.

---

### Step 10: Access the UI

> **Where:** Vast.ai website

Go to Vast.ai Applications page > **Tunnels** (left menu) > Enter `http://localhost:3333` > Click **"+ Create New Tunnel"**.

It will give you a URL like `https://xxx-xxx-xxx.trycloudflare.com`. Open that in your browser.

**Alternative:** The app is also accessible via the external port mapped to 8384 (since we redirected Caddy).

---

## Using the App

1. Open the URL in your browser
2. Drag & drop an image
3. Pick quality (Low/Medium/High)
4. Pick format (GLB for Unity, OBJ, STL, FBX)
5. **Enable "Generate Texture" toggle** (this is why we're on cloud!)
6. Click **Generate 3D Model**
7. Wait ~1-3 min
8. Click **Download**

---

## When You're Done

### Option A: Use again soon (this week)
- **Stop** the instance (pause button)
- Small disk storage fee (~$0.01-0.05/hr)
- Next time: just Start it, open terminal, run `cd /workspace/Hunyuan3D-2 && python app.py`

### Option B: Done for a while
- **Destroy** the instance (trash icon)
- Zero charges
- Next time: follow this entire guide again (~15 min)

---

## Cost Estimates

| Action | Cost |
|--------|------|
| Setup (first time) | ~15 min = ~$0.05 |
| Generate 1 textured model | ~2-5 min = ~$0.01-0.02 |
| Batch of 10 models | ~30-50 min = ~$0.10-0.15 |
| Forgot to stop for 24hr | ~$4.80 |
| Stopped but not destroyed for 24hr | ~$0.50-1.00 |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `cusparse.h: No such file or directory` | Run the `find` + `export` commands from Step 4 |
| `No module named 'custom_rasterizer'` | Redo Step 5 (build texture components) |
| `No space left on device` | Disk too small! Destroy instance, rent one with 80GB+ disk |
| `out of memory` (GPU) | Shouldn't happen on 24GB RTX 3090. Try "Low" quality |
| `Port XXXX is in use` | Do Step 8 (Caddy fix) |
| `total_mem` AttributeError | Already fixed in latest app.py (uses `total_memory`) |
| Paste breaks in terminal | That's why Step 7 uses Notebook, not Terminal |
| Browser says "Generation failed" | Fixed with async polling in latest version |
| "Server may be stuck" warning | Normal during texture generation (3-6 min). Just wait |
| SSL certificate error on Jupyter | Click "Advanced" > "Proceed to site" |
| paste.rs URLs expired | Re-upload files from `D:\Hunyuan3DStudio\` (see Step 7 note) |
| Port 3333 in use | Run `pkill -9 -f "python app.py"` then restart |

---

## Quick Reference

### Where does each step run?

| Step | Jupyter Terminal | Jupyter Notebook |
|------|:---:|:---:|
| Step 3: Clone & Install | X | |
| Step 4: CUDA Headers | X | |
| Step 5: Build Texture | X | |
| Step 6: Create Dirs | X | |
| Step 7: Download App Files | | X |
| Step 8: Fix Caddy | X | |
| Step 9: Start Server | X | |

### Local Files (on your PC)

- `D:\Hunyuan3DStudio\app.py` -- local version (shape-only, RTX 2060)
- `D:\Hunyuan3DStudio\cloud_app.py` -- cloud version (shape + texture)
- `D:\Hunyuan3DStudio\static\index.html` -- the web UI
- `D:\Hunyuan3DStudio\CLOUD_SETUP.md` -- this guide

---

## TODO (Future Improvements)

1. **Face Reduction Slider** - models have ~1.7M faces, too heavy for Unity. Add slider (1K-100K) using `FaceReducer` from `hy3dgen.shapegen.postprocessors`
2. **Turbo Mode Toggle** - 5-step fast preview using `hunyuan3d-dit-v2-mini-turbo` with `--enable_flashvdm`
3. **Seed Control** - currently hardcoded to seed=42, let user set or randomize
4. **Bigger Model Option** - switch between Mini (0.6B) and Standard (1.1B) on cloud
5. ~~**FBX Export Fix**~~ DONE - `libopengl0` in Step 3
6. **Post-Processing Options** - FloaterRemover, DegenerateFaceRemover, texture enhancement
7. **Fast Texture Mode** - reduce diffusion steps, skip light removal, lower resolution views
