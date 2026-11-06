import io
import ipaddress
import os
import posixpath
import socket
import time
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template, request, send_file

from watermark_remover import remove_tiet_watermark_auto

app = Flask(__name__)

MAX_DOWNLOAD_BYTES = 35 * 1024 * 1024  # 35 MB cap
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
}


def validate_public_url(url: str) -> str:
    """Validate that the URL is HTTP(S) and resolves to a public IP address."""
    if not url or not isinstance(url, str):
        raise ValueError("Please provide a valid image URL.")
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http:// and https:// URLs are supported.")
    if not parsed.hostname:
        raise ValueError("Invalid URL hostname.")

    try:
        addr_info = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname: {parsed.hostname}") from exc

    for info in addr_info:
        ip_str = info[4][0]
        ip_obj = ipaddress.ip_address(ip_str)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
        ):
            raise ValueError("Access to private or local network addresses is not allowed.")

    return url


def fetch_image_bytes(url: str) -> tuple[bytes, str]:
    """Download raw original image bytes from a remote URL without any modification."""
    clean_url = validate_public_url(url)
    with requests.get(clean_url, headers=REQUEST_HEADERS, stream=True, timeout=25) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "image/webp")
        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Image exceeds maximum allowed size (35 MB).")
                chunks.append(chunk)
    return b"".join(chunks), content_type


def derive_clean_filename(url: str) -> str:
    """Build a clean '<basename>_cleaned.png' filename from the source URL."""
    path = urlparse(url).path
    base = posixpath.basename(path) or "image"
    stem, _ = os.path.splitext(base)
    safe_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_")) or "image"
    return f"{safe_stem}_cleaned.png"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/preview-source", methods=["GET"])
def preview_source():
    """Proxy the original watermarked image at 100% original bytes for live browser preview."""
    url = request.args.get("url", "").strip()
    try:
        raw_bytes, content_type = fetch_image_bytes(url)
        return Response(
            raw_bytes,
            mimetype=content_type,
            headers={"Cache-Control": "public, max-age=300"},
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/process", methods=["POST", "GET"])
def process_image():
    """
    Download the source image from URL, remove the TIET watermark at 100% native
    pixel resolution without any compression or downscaling, and return a lossless PNG.
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or request.form.get("url") or "").strip()
    else:
        url = request.args.get("url", "").strip()

    try:
        t0 = time.time()
        raw_bytes, _ = fetch_image_bytes(url)

        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        bgr_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr_img is None:
            return jsonify({"error": "Could not decode image from the provided URL."}), 400

        orig_h, orig_w = bgr_img.shape[:2]
        cleaned_bgr = remove_tiet_watermark_auto(bgr_img)

        # Encode as 100% lossless full-resolution PNG (no downscaling, no lossy compression)
        ok, png_buf = cv2.imencode(".png", cleaned_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            return jsonify({"error": "Failed to encode lossless PNG."}), 500

        elapsed = time.time() - t0
        out_filename = derive_clean_filename(url)
        png_bytes = png_buf.tobytes()

        as_attachment = request.args.get("download", "0") == "1"
        resp = send_file(
            io.BytesIO(png_bytes),
            mimetype="image/png",
            as_attachment=as_attachment,
            download_name=out_filename,
        )
        resp.headers["X-Image-Width"] = str(orig_w)
        resp.headers["X-Image-Height"] = str(orig_h)
        resp.headers["X-Process-Seconds"] = f"{elapsed:.2f}"
        resp.headers["X-Download-Filename"] = out_filename
        resp.headers["Access-Control-Expose-Headers"] = (
            "X-Image-Width, X-Image-Height, X-Process-Seconds, X-Download-Filename"
        )
        return resp

    except requests.HTTPError as exc:
        return jsonify({"error": f"Failed to download image from URL (HTTP {exc.response.status_code})."}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
