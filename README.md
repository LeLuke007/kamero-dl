---
title: Kamero DL Watermark Remover
emoji: 🖼️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Kamero-DL — Lossless TIET Watermark Remover Web App

A Python Flask + OpenCV web application that takes a watermarked image URL (such as `https://cdn.kamero.ai/...webp`), downloads the original image on the server, removes the tiled `TIET` watermark at **100% original pixel resolution**, previews Before/After in the browser, and provides a **lossless PNG** download with zero compression or downscaling.

## Project Structure

```text
kamero-dl/
├── app.py                 # Flask server (URL proxy preview + lossless PNG processing API)
├── watermark_remover.py   # Headless OpenCV/NumPy sub-pixel lattice TIET watermark remover
├── templates/
│   └── index.html         # Dark-mode Web UI with Side-by-Side preview, 1:1 zoom & PNG download
├── requirements.txt       # Python dependencies (opencv-python-headless, numpy, flask, requests, gunicorn)
├── Dockerfile             # Ready for Hugging Face Spaces, Render, Koyeb, Cloud Run, Railway
├── render.yaml            # Render.com Blueprint config
└── Procfile               # Process config for PaaS hosts
```

## Run Locally

```bash
cd kamero-dl
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:7860` in your browser.

## Free Hosting & Deployment Guide

### Option 1: Hugging Face Spaces (Recommended — 100% Free 2 vCPU + 16 GB RAM)
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Name your Space (e.g. `kamero-dl`), select **Docker** as the Space SDK, and choose **Blank**.
3. Upload all files inside `kamero-dl/` (including `templates/index.html`, `Dockerfile`, `app.py`, `watermark_remover.py`, `requirements.txt`, `README.md`).
4. Hugging Face will automatically build the Docker container and host your app at `https://<your-username>-kamero-dl.hf.space`.

### Option 2: Render.com (Free Web Service)
1. Push the `kamero-dl/` folder to a GitHub repository.
2. Go to [dashboard.render.com](https://dashboard.render.com) -> **New +** -> **Web Service**.
3. Connect your GitHub repository:
   - **Runtime**: `Python 3` (or `Docker`)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180`
   - **Instance Type**: `Free`
4. Click **Create Web Service**.
