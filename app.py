from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
import threading
from face_detection_utils.utils import load_face_ids_from_db, compare_face_encodings
from database.create_database import add_person_to_db
from face_detection_utils.detect_faces import detect_face

app = Flask(__name__)

# Load face data from the database at startup
face_data = load_face_ids_from_db()


def process_video_stream():
    """
    Function to process the RTSP video stream and display it.
    This will run in a separate thread.
    """
    # Open the RTSP stream
    rtsp_url = "rtsp://localhost:8554/mystream"
    cap = cv2.VideoCapture(rtsp_url)

    # Check if the stream is opened successfully
    if not cap.isOpened():
        print(f"Failed to open RTSP stream: {rtsp_url}")
        return

    # Set buffer size to minimum for lower latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    while True:
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

        for (top, right, bottom, left) in face_locations:
            # Compute the face encoding for the detected face
            face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
            if len(face_encodings) == 0:
                continue  # Skip if no face encoding is found

            encoding = face_encodings[0]
            name, status = compare_face_encodings(encoding, face_data)

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
        return jsonify({
            "message": f"Successfully added {len(image_files)} image(s) for {name} with person_id: {person_id}"
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts the name and status of a person from an uploaded image.
    Expects a POST request with a file under the key 'image'.
    Returns JSON with the name and status of the detected person.
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
        name, status = compare_face_encodings(encoding, face_data)

        if name is None:
            return jsonify({"message": "Unknown person"}), 404

        return jsonify({
            "name": name,
            "status": status
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    # Start the video processing in a separate thread
    video_thread = threading.Thread(target=process_video_stream)
    video_thread.daemon = True  # This makes the thread exit when the main program exits
    video_thread.start()

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)