import cv2
import face_recognition

def detect_face(frame):
    """
    Detects faces in the given frame using HOG method.
    """
    face_locations = face_recognition.face_locations(frame, model="hog")
    return face_locations