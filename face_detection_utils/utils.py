import sqlite3
import numpy as np
import face_recognition


def load_face_ids_from_db():
    """
    Loads all face IDs, names, statuses, and person_ids from the database.
    Groups face encodings by person_id.
    """
    conn = sqlite3.connect('database/db.sqlite3')
    cursor = conn.cursor()

    # Retrieve all rows from the database
    cursor.execute('SELECT person_id, face_id, name, status FROM persons')
    rows = cursor.fetchall()
    conn.close()

    # Group face encodings by person_id
    face_data = {}
    for row in rows:
        person_id, face_id_hex, name, status = row
        stored_encoding = np.frombuffer(bytes.fromhex(face_id_hex), dtype=np.float64)

        if person_id not in face_data:
            face_data[person_id] = {
                'name': name,
                'status': status,
                'encodings': []
            }
        face_data[person_id]['encodings'].append(stored_encoding)

    return face_data



def compare_face_encodings(detected_encoding, face_data):
    """
    Compares the detected face encoding with stored face IDs grouped by person_id.
    Returns the name and status if a match is found, otherwise None.
    """
    for person_id, data in face_data.items():
        name = data['name']
        status = data['status']
        encodings = data['encodings']

        # Compare detected encoding with all encodings for this person
        if face_recognition.compare_faces(encodings, detected_encoding, tolerance=0.5)[0]:
            return name, status

    return None, None