import gc

import cv2
import numpy as np
import tensorflow as tf

from src.model import i3d


# Vierasprocessing functions
def extract_clips(video_path, clip_length=64, step=32):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video file")
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frames.append(frame)
    finally:
        cap.release()  # Release OpenCV VideoCapture
    clips = []
    for i in range(0, len(frames) - clip_length, step):
        clip = np.array(frames[i : i + clip_length])
        if len(clip) == clip_length:
            clips.append(clip)
    del frames  # Clear frames list
    gc.collect()  # Force garbage collection
    return clips


def get_i3d_features(clips):
    features = []
    for clip in clips:
        clip = np.expand_dims(clip, axis=0)
        logits = i3d(tf.constant(clip, dtype=tf.float32))["default"][0]
        features.append(logits.numpy())
        del clip  # Clear clip array
        gc.collect()
    return np.array(features)
