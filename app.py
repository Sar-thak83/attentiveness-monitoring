import cv2
from flask import Flask, render_template, Response, redirect, url_for
import mediapipe as mp
import datetime
import csv
from utils import mediapipe_detection, predict_attentiveness, draw_styled_landmarks
import io
import pandas as pd
import matplotlib.pyplot as plt
import os

# Customize these based on your setup
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Flask app setup
app = Flask(__name__)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

# Initialize global variables
recording = False
timestamps = []
attentiveness_levels = []
detect = False

# Route for the main web page
@app.route('/')
def index():
    return render_template('index.html', recording=recording, timestamps=timestamps, attentiveness_levels=attentiveness_levels)

# Route for video streaming
def generate_frames():
    global recording, timestamps, attentiveness_levels, detect

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_frame = frame

        if detect:
            # Process frame with MediaPipe (including attentiveness prediction and visualization)
            image, results = mediapipe_detection(frame, mp_holistic)
            prediction = predict_attentiveness(results)

            # Draw styled landmarks on the frame with specified font scale
            image = draw_styled_landmarks(image, results, font_scale=2)

            # Capture timestamps and attentiveness levels if recording
            if recording:
                timestamps.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                attentiveness_levels.append(prediction)

            out_frame = image

        # Encode frame as JPEG bytestream
        encoded = cv2.imencode('.jpg', out_frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

# Start recording button handler
@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording,detect
    recording = True
    detect = True
    return redirect(url_for('index'))

# Stop recording button handler
@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, timestamps, attentiveness_levels, detect

    recording = False
    detect = False

    # Save timestamps and attentiveness levels to a local CSV file
    if timestamps and attentiveness_levels:
        file_exists = os.path.exists("attentiveness_data.csv")
        with open("attentiveness_data.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Timestamp", "Prediction"])
            for timestamp, attentiveness in zip(timestamps, attentiveness_levels):
                writer.writerow([timestamp, attentiveness])

    timestamps = []
    attentiveness_levels = []

    return redirect(url_for('index'))

@app.route('/download_report')
def download_report():
    global timestamps, attentiveness_levels

    if os.path.exists('attentiveness_data.csv'):
        try:
            df = pd.read_csv('attentiveness_data.csv')
            timestamps = df['Timestamp'].tolist()
            attentiveness_levels = df['Prediction'].tolist()
        except Exception as e:
            return f"Error reading data file: {e}", 500
    elif not timestamps or not attentiveness_levels:
        return "No data available for report generation.", 404

    image_bytes = generate_report_image(timestamps, attentiveness_levels)

    response = Response(image_bytes, mimetype='image/png')
    response.headers['Content-Disposition'] = 'attachment; filename=attentiveness_report.png'
    return response

def generate_report_image(timestamps, attentiveness_levels):
    """
    Generates a line graph of attentiveness over time and returns the image data as bytes.
    """

    # Build dataframe from provided lists
    df = pd.DataFrame({'Timestamp': timestamps, 'Prediction': attentiveness_levels})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    # Group by 1-minute intervals and count occurrences of each prediction
    resampled = df.groupby([pd.Grouper(freq='T'), 'Prediction']).size().unstack(fill_value=0)

    # Create new figure and plot
    fig, ax = plt.subplots(figsize=(12, 6))
    resampled.plot(kind='line', marker='o', ax=ax)
    ax.set_title('Attentiveness Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.grid(True)
    ax.legend(title='Attentiveness')

    # Save the plot as a PNG image in memory
    buffer = io.BytesIO()
    fig.canvas.print_png(buffer)
    buffer.seek(0)
    plt.close(fig)

    return buffer.read()

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
