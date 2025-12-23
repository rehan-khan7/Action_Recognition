import glob
import logging
import os
import time

import requests
from flask import Flask, flash, redirect, render_template, request, url_for

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
API_URL = "http://127.0.0.1:8000/recognize_action"
UPLOAD_FOLDER = "static/uploads"

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def clean_old_files():
    """Remove files older than 5 minutes from the upload folder."""
    current_time = time.time()
    for file_path in glob.glob(os.path.join(UPLOAD_FOLDER, "*")):
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > 300:  # 5 minutes
                try:
                    os.remove(file_path)
                    logger.info(f"Removed old file: {file_path}")
                except Exception as e:  # pylint: disable=broad-except
                    logger.error(f"Error removing file {file_path}: {e}")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Clean old files
        clean_old_files()

        # Check if a video file is uploaded
        if "video" not in request.files:
            flash("No video file uploaded")
            return redirect(url_for("index"))

        video = request.files["video"]
        if video.filename == "":
            flash("No video selected")
            return redirect(url_for("index"))

        # Save video with a unique filename
        timestamp = int(time.time())
        video_filename = f"{timestamp}_{video.filename}"
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        video.save(video_path)
        logger.info(f"Saved video to: {video_path}")

        # Send video to FastAPI backend
        try:
            with open(video_path, "rb") as f:
                files = {"video": (video.filename, f, "video/mp4")}
                response = requests.post(API_URL, files=files, timeout=60)

            if response.status_code == 200:
                actions = response.json().get("actions", [])
                # Pass the video filename to the template
                return render_template(
                    "index.html", actions=actions, video_file=video_filename
                )
        except Exception as e:  # pylint: disable=broad-except
            flash(f"Error: {str(e)}")
            # Remove video on error
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Removed video on error: {video_path}")
        return redirect(url_for("index"))

    # Clean old files on GET request
    clean_old_files()
    return render_template("index.html", actions=None, video_file=None)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
