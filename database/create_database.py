import os
import sqlite3
import face_recognition
import uuid

# Define the default database path
DB_PATH = 'database/db.sqlite3'


def init_db(db_path=DB_PATH):
    """
    Initializes the database by ensuring the directory exists and
    creating the 'persons' table if it does not already exist.
    """
    # Create database directory if it doesn't exist
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    # Connect to the database and create the table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS persons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT NOT NULL,
        face_id TEXT NOT NULL,
        name TEXT NOT NULL,
        status TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()


def extract_face_encoding(image_path):
    """
    Extracts the face encoding from an image file.
    Returns the encoding as a hexadecimal string.
    """
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if len(face_encodings) == 0:
        raise ValueError(f"No face detected in {image_path}")

    # Convert the encoding to a hexadecimal string for storage
    return face_encodings[0].tobytes().hex()


def add_person_to_db(image_files, name, status, person_id=None, db_path=DB_PATH):
    """
    Adds a person's face encoding(s), name, and status to the database.
    :param image_files: A list of file-like objects (e.g., uploaded files from Flask).
    :param name: Name of the person.
    :param status: Status of the person ('blacklisted' or 'not_blacklisted').
    :param person_id: Unique identifier for the person. If None, a new ID is generated.
    :param db_path: Path to the SQLite database file.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Generate a new person_id if not provided
    if person_id is None:
        person_id = str(uuid.uuid4())

    # Process each uploaded file
    for image_file in image_files:
        try:
            # Save the uploaded file temporarily
            temp_image_path = f"temp_{uuid.uuid4()}.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(image_file.read())

            # Extract face encoding
            face_id = extract_face_encoding(temp_image_path)

            # Insert the person's data into the database
            cursor.execute('INSERT INTO persons (person_id, face_id, name, status) VALUES (?, ?, ?, ?)',
                           (person_id, face_id, name, status))
            print(f"Added encoding for {image_file.filename} to the database with person_id: {person_id}.")

            # Clean up the temporary file
            os.remove(temp_image_path)

        except ValueError as e:
            print(f"Skipping {image_file.filename}: {e}")
        except Exception as e:
            print(f"Error processing {image_file.filename}: {e}")

    conn.commit()
    conn.close()
    return person_id
