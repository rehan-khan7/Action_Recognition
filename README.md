# Action Recognition in Sports Videos

This project provides a web-based system for recognizing actions in sports videos, such as "shooting," "passing," and "dribbling" in basketball. It uses a **FastAPI** backend to process videos with a pre trained I3D model and an LSTM model, and a **Flask** frontend to allow users to upload videos and view results through a web interface.

## Features

- Upload sports videos (MP4 or AVI) via a web interface.
- Detect actions using I3D for feature extraction and LSTM for sequence modeling.
- Display recognized actions and the uploaded video in the browser.
- Memory management to prevent GPU and disk space leaks.
- Logging for debugging and monitoring.

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU (e.g., RTX 4080 SUPER) with CUDA and cuDNN for TensorFlow
- FFmpeg installed for video processing (`sudo apt-get install ffmpeg` on Ubuntu)
- A trained LSTM model weights file (`lstm_model_weights.h5`) or a dataset to train the model (e.g., UCF101)

## Installation

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd action_recognition
   ```

2. Create a conda Environment:

```bash
conda create -n conda_env python=3.10
conda activate conda_env
```

3. Install Dependencies:

```bash
pip install -r requirements.txt
```

4. Install FFmpeg (if not already installed)

5 .Running the Application

6. Start the FastAPI Backend:

```bash
uvicorn api:app --reload --port 8000
```

The backend will run on http://127.0.0.1:8000.

Start the Flask Frontend:

```bash
python app.py
```

The frontend will run on http://127.0.0.1:5000.

- Access the Web Interface:

Open http://127.0.0.1:5000 in your browser.
Upload a sports video (MP4 recommended, AVI supported) to analyze actions.
View the recognized actions (e.g., "shooting," "passing," "dribbling") and the uploaded video.

# Usage

## Uploading a Video:

Use the web interface to select a video file (e.g., test_video.mp4).
Click "Upload and Analyze" to process the video.
The results will show the top recognized actions and the video in a player.

## Sample Video:

Download a video from the UCF101 [dataset](http://crcv.ucf.edu/data/UCF101.php), e.g., `v_Basketball_g01_c01.avi`.
Convert to MP4 for browser compatibility:

```bash
ffmpeg -i v_Basketball_g01_c01.avi -c:v libx264 -c:a aac -strict -2 test_video.mp4
```

## Testing with Postman (Optional):

- Send a POST request to `http://127.0.0.1:8000/recognize_action` with a `form-data` body:
- Key: `video`, Value: Select your video file.

Check the JSON response for actions.

# Troubleshooting

## Video Not Displaying:

- Cause: The video file may be deleted before rendering or in an unsupported format.

- Fix: Ensure the video is MP4 with H.264 codec:

```bash
ffmpeg -i input.avi -c:v libx264 -c:a aac -strict -2 output.mp4
```

Check Flask logs for file creation/deletion:INFO:**main**:Saved video to: static/uploads/...

Inspect the browser’s Network tab for the video URL (e.g., http://127.0.0.1:5000/static/uploads/1634567890_test_video.mp4).

## "No valid clips extracted" Error:

- Cause: Video is too short (<64 frames, ~2 seconds at 30 fps).

- Fix:
  Use a longer video or reduce clip_length in api.py:clips = extract_clips(video_path, clip_length=32, step=16)

- Verify video readability:

```python
cap = cv2.VideoCapture("test_video.mp4")
print(cap.isOpened()) # Should print True
cap.release()
```

## Memory Issues:

- Monitor GPU memory with nvidia-smi.
- Check disk space in static/uploads and temp_uploads.
- The code includes memory cleanup (tf.keras.backend.clear_session(), gc.collect(), file deletion).

# Notes

- Model: The LSTM model must be trained or loaded with weights for accurate predictions. Update ACTION_LABELS to match your dataset.

- Video Format: MP4 with H.264 is recommended for browser compatibility. Convert AVI files using FFmpeg.

- Performance: For large videos, reduce clip_length or optimize preprocessing.

- Production: Use Gunicorn for Flask (gunicorn -w 4 app:app -b 0.0.0.0:5000) and Uvicorn workers for FastAPI. Consider a reverse proxy (e.g., Nginx).

# Acknowledgments

- UCF101 Dataset for sample videos.
- TensorFlow Hub for the I3D model.
- FastAPI and Flask for backend and frontend frameworks.
