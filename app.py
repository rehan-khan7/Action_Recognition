import glob
import logging
import os
import time

import requests
from flask import Flask, flash, redirect, render_template, request, url_for

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

SERVICE_ENDPOINT = "http://127.0.0.1:8000/recognize_action"
UPLOAD_DIR = "static/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


def remove_stale_uploads(retention_seconds=300):
    now = time.time()
    for path in glob.glob(os.path.join(UPLOAD_DIR, "*")):
        if os.path.isfile(path) and now - os.path.getmtime(path) > retention_seconds:
            try:
                os.remove(path)
                log.info("Deleted stale upload: %s", path)
            except Exception as exc:  # pylint: disable=broad-except
                log.error("Could not delete %s: %s", path, exc)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        remove_stale_uploads()

        if "video" not in request.files:
            flash("No video file uploaded")
            return redirect(url_for("index"))

        file_storage = request.files["video"]
        if file_storage.filename == "":
            flash("No video selected")
            return redirect(url_for("index"))

        ts = int(time.time())
        saved_name = f"{ts}_{file_storage.filename}"
        saved_path = os.path.join(UPLOAD_DIR, saved_name)
        file_storage.save(saved_path)
        log.info("Saved upload: %s", saved_path)

        try:
            with open(saved_path, "rb") as fh:
                files = {"video": (file_storage.filename, fh, "video/mp4")}
                resp = requests.post(SERVICE_ENDPOINT, files=files, timeout=60)

            if resp.status_code == 200:
                actions = resp.json().get("actions", [])
                return render_template(
                    "index.html", actions=actions, video_file=saved_name
                )
        except Exception as exc:  # pylint: disable=broad-except
            flash(f"Error: {exc}")
            if os.path.exists(saved_path):
                try:
                    os.remove(saved_path)
                    log.info("Removed upload after error: %s", saved_path)
                except Exception as rm_exc:
                    log.error("Failed removing file %s: %s", saved_path, rm_exc)

        return redirect(url_for("index"))

    remove_stale_uploads()
    return render_template("index.html", actions=None, video_file=None)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
