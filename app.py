import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
import threading
import time
import sqlite3
from flask_cors import CORS
from face_detection_utils.utils import compare_face_encodings
from database.create_database import add_person_to_db, init_db
from face_detection_utils.detect_faces import detect_face

app = Flask(__name__)
CORS(app)
# Initialize the database synchronously at startup
init_db()

# Global variables with locks for thread safety
face_data_lock = threading.Lock()
face_data = {}
last_db_update_time = 0


def load_face_ids_from_db_direct():
    """
    Directly loads face data from the database.
    Returns a dictionary mapping face_id (hex string) to (name, status).
    """
    face_data_dict = {}
    try:
        db_path = 'database/db.sqlite3'  # Adjust path if needed
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fetch all face IDs, names, and statuses from the database
        cursor.execute('SELECT face_id, name, status FROM persons')
        rows = cursor.fetchall()

        for face_id_hex, name, status in rows:
            # Convert hex string to numpy array
            face_id_bytes = bytes.fromhex(face_id_hex)
            face_encoding = np.frombuffer(face_id_bytes, dtype=np.float64)

            # Store the face encoding and associated info
            face_data_dict[face_id_hex] = (face_encoding, name, status)

        conn.close()
        print(f"Successfully loaded {len(face_data_dict)} face entries from database")
        return face_data_dict

    except Exception as e:
        print(f"Error loading face data: {e}")
        return {}


def check_and_reload_face_data():
    """
    Checks if the database has been updated and reloads face data if necessary.
    """
    global face_data, last_db_update_time

    try:
        db_path = 'database/db.sqlite3'
        if os.path.exists(db_path):
            current_mtime = os.path.getmtime(db_path)

            # If file was modified since last check
            if current_mtime > last_db_update_time:
                with face_data_lock:
                    face_data = load_face_ids_from_db_direct()
                    last_db_update_time = current_mtime
                    print(f"Face data reloaded at {time.strftime('%H:%M:%S')}. Total entries: {len(face_data)}")
                    return True
    except Exception as e:
        print(f"Error checking database update time: {e}")

    return False


def compare_faces_directly(face_encoding, face_data_dict):
    """
    Directly compares a face encoding with all encodings in the database.
    Returns the name and status of the closest match, or (None, None) if no match is found.
    """
    if not face_data_dict:
        return None, None

    # Set a threshold for face recognition
    threshold = 0.6  # Lower is more strict, higher is more lenient

    best_match = None
    min_distance = float('inf')

    # Compare with all faces in the database
    for face_id_hex, (encoding, name, status) in face_data_dict.items():
        # Calculate face distance
        face_distance = face_recognition.face_distance([encoding], face_encoding)[0]

        # Update best match if this is closer
        if face_distance < min_distance:
            min_distance = face_distance
            best_match = (name, status)

    # If the best match is too far, consider it unknown
    if min_distance > threshold:
        return None, None

    return best_match


def process_video_stream():
    """
    Function to process the RTSP video stream and display it.
    This will run in a separate thread.
    """
    global face_data

    # Open the RTSP stream
    rtsp_url = "rtsp://192.168.1.121:8554/mystream"
    cap = cv2.VideoCapture(rtsp_url)

    # Check if the stream is opened successfully
    if not cap.isOpened():
        print(f"Failed to open RTSP stream: {rtsp_url}")
        return

    # Set buffer size to minimum for lower latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    check_interval = 50  # Check for DB updates every 50 frames

    while True:
        # Check if database has been updated periodically
        if frame_count % check_interval == 0:
            check_and_reload_face_data()

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from RTSP stream")
            # Try to reconnect
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print("Failed to reconnect to RTSP stream")
                break
            continue

        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame
            continue

        # Detect faces in the frame using Haar Cascade
        face_locations = detect_face(frame)

        # Convert the frame to RGB for face encoding
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get a copy of the current face data
        with face_data_lock:
            current_face_data = face_data.copy()

        for (top, right, bottom, left) in face_locations:
            # Compute the face encoding for the detected face
            face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
            if len(face_encodings) == 0:
                continue  # Skip if no face encoding is found

            encoding = face_encodings[0]
            name, status = compare_faces_directly(encoding, current_face_data)

            # Determine the label and color based on the status
            if status == 'blacklisted':
                label = f"ALERT: {name} (Blacklisted)"
                color = (0, 0, 255)  # Red
            elif status == 'not_blacklisted':
                label = f"{name} (Not Blacklisted)"
                color = (0, 255, 0)  # Green
            else:
                label = "Unknown"
                color = (255, 255, 255)  # White

            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display the frame
        cv2.imshow('Video Stream', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


@app.route('/upload_images', methods=['POST'])
def upload_images():
    name = request.form.get('name')
    status = request.form.get('status')

    # Debugging prints
    print("Form Data:", request.form)
    print("Files Data:", request.files)

    if not name or not status:
        return jsonify({"error": "Missing 'name' or 'status' in request"}), 400

    image_files = request.files.getlist('images')

    # Debugging prints
    print("Extracted Files:", image_files)

    if not image_files or image_files[0].filename == '':
        return jsonify({"error": "No images provided"}), 400

    try:
        person_id = add_person_to_db(image_files=image_files, name=name, status=status)

        # Force reload face data immediately
        with face_data_lock:
            global face_data
            face_data = load_face_ids_from_db_direct()
            print(f"Face data reloaded after adding {name}. Total entries: {len(face_data)}")

        return jsonify({
            "message": f"Successfully added {len(image_files)} image(s) for {name} with person_id: {person_id}"
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/reload', methods=['GET'])
def reload_face_data_endpoint():
    """
    Endpoint to manually reload face data.
    """
    try:
        with face_data_lock:
            global face_data
            face_data = load_face_ids_from_db_direct()
        return jsonify({
            "message": "Face data reloaded successfully",
            "entries": len(face_data)
        }), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts the name and status of a person from an uploaded image.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Read the uploaded image file
    image_file = request.files['image']
    try:
        # Convert the file to a numpy array
        image_bytes = image_file.read()
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Detect faces in the image
        face_locations = detect_face(frame)

        if len(face_locations) == 0:
            return jsonify({"message": "No face detected in the image"}), 404

        # Process the first detected face
        top, right, bottom, left = face_locations[0]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])

        if len(face_encodings) == 0:
            return jsonify({"message": "No face encoding found"}), 404

        encoding = face_encodings[0]

        # Get a copy of the current face data
        with face_data_lock:
            current_face_data = face_data.copy()

        name, status = compare_faces_directly(encoding, current_face_data)

        if name is None:
            return jsonify({"message": "Unknown person"}), 404

        return jsonify({
            "name": name,
            "status": status
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    # Initial load of face data
    face_data = load_face_ids_from_db_direct()
    last_db_update_time = os.path.getmtime('database/db.sqlite3') if os.path.exists('database/db.sqlite3') else 0

    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        # Start the video processing in a separate thread
        video_thread = threading.Thread(target=process_video_stream)
        video_thread.daemon = True
        video_thread.start()

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)