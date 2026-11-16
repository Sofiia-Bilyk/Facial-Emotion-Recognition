# Live facial expression recognition

## Run on this computer

From PowerShell, use the existing Anaconda Python 3.12 environment:

```powershell
Set-Location 'C:\Coding\ML\Facial Emotion Recognition'
& "$env:USERPROFILE\anaconda3\python.exe" -B -m flask --app live_detection.app run --host 127.0.0.1 --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in Chrome or Edge. Click **Start Camera** and allow camera access. Check that the preview stays live and the prediction panel shows an expression, emoji, and time in milliseconds. Move out of frame to check that **No face detected** appears and the emoji clears. Click **Stop Camera** to release the webcam and reset the display. Press **Ctrl+C** in PowerShell to stop Flask.

## Set up an isolated environment

```powershell
Set-Location 'C:\Coding\ML\Facial Emotion Recognition'
& "$env:USERPROFILE\anaconda3\python.exe" -m venv .\live_detection\.venv
& .\live_detection\.venv\Scripts\python.exe -m pip install -r .\live_detection\requirements.txt
& .\live_detection\.venv\Scripts\python.exe -B -m flask --app live_detection.app run --host 127.0.0.1 --port 5000
```

## Files

- `app.py` serves the page and the local JPEG prediction endpoint.
- `face_detection.py` finds and crops the largest frontal face.
- `inference.py` loads the root checkpoint and predicts an expression from a grayscale face crop.
- `templates/index.html` defines the webcam and prediction panels.
- `static/styles.css` lays out the panels.
- `static/camera.js` controls the camera and processes one snapshot at a time. Processing time runs from immediately before capture to the next paint opportunity after the expression and emoji update. The pause between snapshots is excluded.
- `requirements.txt` lists the tested Python package versions.
