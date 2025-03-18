import os
import cv2
import numpy as np
import face_recognition
import threading
import time
import sqlite3
import streamlit as st
import queue
import gc  # Added for garbage collection
from face_detection_utils.utils import compare_face_encodings
from database.create_database import add_person_to_db, init_db
from face_detection_utils.detect_faces import detect_face
from utils.constants import RSTP_URL

# Initialize the database synchronously at startup
init_db()

# Global variables with locks for thread safety
face_data_lock = threading.Lock()
upload_lock = threading.Lock()  # Lock for upload and prediction operations
face_data = {}
last_db_update_time = 0

# Create a session state variable to track if stream is running
if 'stream_running' not in st.session_state:
    st.session_state.stream_running = False


def cleanup_resources():
    """Clean up memory resources after intensive operations"""
    cv2.destroyAllWindows()
    gc.collect()


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


class VideoStreamHandler:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)  # Limit queue size to 1
        self.thread = None
        self.cap = None

    def process_video_stream(self):
        """
        Function to process the RTSP video stream and display it.
        This will run in a separate thread.
        """
        global face_data

        try:
            # Open the RTSP stream
            self.cap = cv2.VideoCapture(self.rtsp_url)

            # Check if the stream is opened successfully
            if not self.cap.isOpened():
                print(f"Failed to open RTSP stream: {self.rtsp_url}")
                return

            # Set buffer size to minimum for lower latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            frame_count = 0
            check_interval = 50  # Check for DB updates every 50 frames

            while not self.stop_event.is_set():
                # Check if database has been updated periodically
                if frame_count % check_interval == 0:
                    check_and_reload_face_data()

                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame from RTSP stream")
                    # Try to reconnect
                    self.cap.release()
                    time.sleep(1)  # Wait before reconnecting
                    self.cap = cv2.VideoCapture(self.rtsp_url)
                    if not self.cap.isOpened():
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

                # Clear the queue to keep only the latest frame
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    # If queue is full, remove the old frame and add the new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame, block=False)
                    except (queue.Empty, queue.Full):
                        pass

        except Exception as e:
            print(f"Error in video stream processing: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            # Release resources
            if self.cap is not None:
                self.cap.release()
            cleanup_resources()  # Added cleanup

    def start_stream(self):
        """
        Start the video stream in a separate thread.
        """
        if not self.thread or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self.process_video_stream)
            self.thread.daemon = True
            self.thread.start()
            st.session_state.stream_running = True
            print("Stream started in a new thread")

    def stop_stream(self):
        """
        Stop the video stream.
        """
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)  # Wait for thread to finish with timeout
            print("Stream stopped")

        # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        st.session_state.stream_running = False

        # Ensure camera is released
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cleanup_resources()  # Added cleanup


def convert_uploaded_files(uploaded_files):
    """
    Convert Streamlit UploadedFile objects to a format compatible with add_person_to_db
    """
    converted_files = []
    for uploaded_file in uploaded_files:
        # Create a file-like object that mimics the original file interface
        file_bytes = uploaded_file.read()

        # Reset file pointer for future reads
        uploaded_file.seek(0)

        file_like = type('FileWrapper', (), {
            'read': lambda: file_bytes,
            'name': uploaded_file.name,
            'filename': uploaded_file.name  # Add filename attribute
        })()

        converted_files.append(file_like)

    return converted_files


def main():
    # Declare global variables used within the function
    global face_data

    # Set page title and layout
    st.set_page_config(page_title="Face Recognition System", layout="wide")
    st.title("Face Recognition System")

    # Initialize the database
    init_db()

    # Sidebar navigation
    menu = st.sidebar.selectbox("Menu",
                                ["Video Stream", "Upload Images", "Predict Face", "Manage Database"]
                                )

    # RTSP Stream URL configuration
    rtsp_url = st.sidebar.text_input(
        "RTSP Stream URL",
        value="rtsp://192.168.1.121:8554/mystream"
    )

    # Initialize video stream handler (singleton instance stored in session state)
    if 'video_stream_handler' not in st.session_state:
        st.session_state.video_stream_handler = VideoStreamHandler(rtsp_url)
    else:
        # Update URL if changed
        st.session_state.video_stream_handler.rtsp_url = rtsp_url

    video_stream_handler = st.session_state.video_stream_handler

    # Display stream status in sidebar
    stream_status = "Running" if st.session_state.stream_running else "Stopped"
    st.sidebar.info(f"Stream Status: {stream_status}")

    if menu == "Video Stream":
        st.header("Real-Time Video Stream")

        # Start/Stop Stream Buttons
        col1, col2 = st.columns(2)
        with col1:
            start_stream = st.button("Start Stream")
        with col2:
            stop_stream = st.button("Stop Stream")

        # Video display placeholder
        video_placeholder = st.empty()

        if start_stream:
            video_stream_handler.start_stream()
            st.success("Stream started.")

        if stop_stream:
            video_stream_handler.stop_stream()
            st.success("Stream stopped.")

        # Continuously update the video frame if the stream is running
        if st.session_state.stream_running:
            update_placeholder = st.empty()
            while st.session_state.stream_running:
                try:
                    frame = video_stream_handler.frame_queue.get(timeout=0.5)
                    video_placeholder.image(frame, channels="BGR")
                    update_placeholder.text(f"Last frame update: {time.strftime('%H:%M:%S')}")
                except queue.Empty:
                    time.sleep(0.1)
                    continue
                except Exception as e:
                    st.error(f"Error displaying frame: {e}")
                    break

    elif menu == "Upload Images":
        st.header("Upload Person Images")

        # Form for uploading images
        with st.form("upload_images_form"):
            name = st.text_input("Person Name")
            status = st.selectbox("Status", ["not_blacklisted", "blacklisted"])
            uploaded_files = st.file_uploader(
                "Choose images",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True
            )
            submit_button = st.form_submit_button("Upload")

            if submit_button:
                if not name or not uploaded_files:
                    st.error("Please provide a name and select images.")
                else:
                    with st.spinner("Uploading images..."):
                        try:
                            # Check if stream is running and temporarily stop it
                            stream_was_running = st.session_state.stream_running
                            if stream_was_running:
                                st.info("Temporarily stopping stream for database update...")
                                video_stream_handler.stop_stream()

                            # Acquire lock for upload operation
                            with upload_lock:
                                # Convert uploaded files to a format compatible with add_person_to_db
                                converted_files = convert_uploaded_files(uploaded_files)
                                # Add person to database
                                person_id = add_person_to_db(
                                    image_files=converted_files,
                                    name=name,
                                    status=status
                                )

                                # Force reload of face data
                                with face_data_lock:
                                    face_data = load_face_ids_from_db_direct()

                            # Restart stream if it was running before
                            if stream_was_running:
                                video_stream_handler.start_stream()
                                st.info("Stream restarted.")

                            st.success(
                                f"Successfully added {len(uploaded_files)} image(s) "
                                f"for {name} with person_id: {person_id}"
                            )
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            # Optional: print full traceback for debugging
                            import traceback
                            st.error(traceback.format_exc())

                            # Ensure stream is restarted if it was running
                            if 'stream_was_running' in locals() and stream_was_running:
                                video_stream_handler.start_stream()
                        finally:
                            # Clean up resources
                            cleanup_resources()

    elif menu == "Predict Face":
        st.header("Face Prediction")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png']
        )

        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image")

            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    try:
                        # Check if stream is running and temporarily stop it
                        stream_was_running = st.session_state.stream_running
                        if stream_was_running:
                            st.info("Temporarily stopping stream for face prediction...")
                            video_stream_handler.stop_stream()

                        # Use the upload lock for prediction too since it involves similar resources
                        with upload_lock:
                            # Read the uploaded image file
                            image_bytes = uploaded_file.read()
                            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
                            frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

                            if frame is None:
                                st.error("Invalid image format")
                                return

                            # Detect faces in the image
                            face_locations = detect_face(frame)

                            if len(face_locations) == 0:
                                st.warning("No face detected in the image")
                                return

                            # Process the first detected face
                            top, right, bottom, left = face_locations[0]
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            # Immediately process the face encoding
                            face_encodings = face_recognition.face_encodings(
                                rgb_frame, [(top, right, bottom, left)]
                            )

                            if len(face_encodings) == 0:
                                st.warning("No face encoding found")
                                return

                            encoding = face_encodings[0]

                            # Load face data with lock
                            with face_data_lock:
                                current_face_data = face_data.copy() if face_data else load_face_ids_from_db_direct()

                            name, status = compare_faces_directly(encoding, current_face_data)

                            # Clear references to help garbage collection
                            rgb_frame = None
                            face_encodings = None

                            if name is None:
                                st.warning("Unknown person")
                            else:
                                st.success(f"Identified: {name}")
                                st.info(f"Status: {status}")

                                # Highlight the detected face
                                cv2.rectangle(
                                    frame,
                                    (left, top),
                                    (right, bottom),
                                    (0, 255, 0),
                                    2
                                )
                                st.image(frame, channels="BGR", caption="Detected Face")

                        # Restart stream if it was running before
                        if stream_was_running:
                            video_stream_handler.start_stream()
                            st.info("Stream restarted.")

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())

                        # Ensure stream is restarted if it was running
                        if 'stream_was_running' in locals() and stream_was_running:
                            video_stream_handler.start_stream()
                    finally:
                        # Clean up resources regardless of success or failure
                        cleanup_resources()

    elif menu == "Manage Database":
        st.header("Database Management")

        if st.button("Reload Face Data"):
            with st.spinner("Reloading face data..."):
                # Check if stream is running and temporarily stop it
                stream_was_running = st.session_state.stream_running
                if stream_was_running:
                    st.info("Temporarily stopping stream for database reload...")
                    video_stream_handler.stop_stream()

                # Reload face data with lock
                with face_data_lock:
                    face_data = load_face_ids_from_db_direct()

                # Restart stream if it was running before
                if stream_was_running:
                    video_stream_handler.start_stream()
                    st.info("Stream restarted.")

                st.success(f"Face data reloaded. Total entries: {len(face_data)}")


if __name__ == "__main__":
    main()