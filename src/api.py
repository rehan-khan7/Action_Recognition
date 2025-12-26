import gc
import logging
import os
import shutil

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.model import build_lstm_model
from src.video_processing import extract_clips, get_i3d_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
lstm_model = build_lstm_model(num_classes=10)
# lstm_model.load_weights("lstm_model_weights.h5")  # Uncomment if weights are available


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Action Recognition API. Use POST /recognize_action to upload a video."
    }


@app.post("/recognize_action")
async def recognize_action(video: UploadFile = File(...)):
    video_path = "uploaded_video.mp4"
    try:
        # Save uploaded video
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        logger.info(f"Processing video: {video.filename}")

        # Extract clips
        clips = extract_clips(video_path)
        if not clips:
            raise HTTPException(
                status_code=400, detail="No valid clips extracted from video"
            )

        # Get I3D features
        features = get_i3d_features(clips)
        del clips  # Clear clips array
        gc.collect()

        # Predict actions
        sequence = np.array(features)
        prediction = lstm_model.predict(np.expand_dims(sequence, axis=0))
        actions = decode_predictions(prediction)

        # Clear memory
        del features, sequence, prediction
        tf.keras.backend.clear_session()  # Clear TensorFlow session
        gc.collect()
        logger.info("Memory cleared after prediction")

        return {"actions": actions}
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing video: {str(e)}"
        ) from e
    finally:
        # Remove temporary file
        if os.path.exists(video_path):
            os.remove(video_path)
            logger.info(f"Removed temporary file: {video_path}")


def decode_predictions(prediction):
    ACTION_LABELS = [
        "shooting",
        "dribbling",
        "passing",
        "dunking",
        "rebounding",
        "blocking",
        "stealing",
        "assisting",
        "jumping",
        "running",
    ]
    top_actions = [ACTION_LABELS[i] for i in np.argsort(prediction[0])[-3:][::-1]]
    return top_actions
