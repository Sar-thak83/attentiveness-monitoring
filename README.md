# Attentiveness Monitoring System

Real-time web application that monitors user attentiveness using MediaPipe and Flask by detecting head pose and eye closure.

## Features

- Real-time video streaming with landmarks
- Head pose detection (yaw/pitch angles)
- Eye closure detection (EAR)
- Recording sessions with timestamps
- CSV data export
- Visual attentiveness reports
- Green (Attentive) / Red (Not Attentive) indicators

## Requirements

- Python 3.0+
- OpenCV, MediaPipe, Flask, Pandas, Matplotlib, NumPy

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   python app.py
   ```

3. Open browser: `http://localhost:5000`

## Usage

- **Enable Detection**: Start viewing video feed with landmarks
- **Start Recording**: Begin tracking attentiveness
- **Stop Recording**: Save data to CSV
- **Download Report**: Generate attentiveness graph

## Detection Algorithm

Attentive = Head centered (±25° yaw, ±20° pitch) AND eyes open (EAR > 0.2)

**Thresholds** (adjust in `utils.py`):
```python
head_yaw_threshold = 25
head_pitch_threshold = 20
eye_aspect_ratio_threshold = 0.2
```

## Output Files

- `attentiveness_data.csv` - Timestamps and predictions
- `attentiveness_report.png` - Attentiveness trends graph

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Always "Attentive" | Lower thresholds in `predict_attentiveness()` |
| Always "Not attentive" | Increase thresholds or improve lighting |
| Poor performance | Reduce VIDEO_WIDTH/HEIGHT or set model_complexity=0 |
| No data saved | Ensure "Start Recording" was clicked |

## Project Structure

```
├── app.py              # Flask application
├── utils.py            # Detection functions
├── templates/index.html # Web interface
└── README.md           # This file
```

## Resources

- [MediaPipe Documentation](https://mediapipe.dev/)
- [Flask Documentation](https://flask.palletsprojects.com/)