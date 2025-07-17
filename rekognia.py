import face_recognition
import cv2
import os
import pickle
import numpy as np

DB_DIR = "database"

def load_database():
    database = {}
    for filename in os.listdir(DB_DIR):
        if filename.endswith(".pkl"):
            with open(os.path.join(DB_DIR, filename), 'rb') as f:
                database[filename[:-4]] = pickle.load(f)
    return database

def save_face(name, encoding):
    with open(os.path.join(DB_DIR, f"{name}.pkl"), 'wb') as f:
        pickle.dump(encoding, f)

def detect_face(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        print("❌ No face found.")
        return None
    return encodings[0]

def add_face(image_path, name):
    encoding = detect_face(image_path)
    if encoding is not None:
        save_face(name, encoding)
        print(f"✅ Saved face as '{name}'.")

def compare_face(image_path):
    unknown_encoding = detect_face(image_path)
    if unknown_encoding is None:
        return

    database = load_database()
    if not database:
        print("⚠️ No faces in database.")
        return

    names = list(database.keys())
    encodings = list(database.values())

    results = face_recognition.compare_faces(encodings, unknown_encoding)
    distances = face_recognition.face_distance(encodings, unknown_encoding)

    if True in results:
        match_index = results.index(True)
        print(f"🎯 Exact match found: {names[match_index]}")
    else:
        closest_index = np.argmin(distances)
        print(f"🤏 Closest match: {names[closest_index]} (distance: {distances[closest_index]:.2f})")

def main():
    while True:
        print("\n=== Facial Recognition Menu ===")
        print("1. Add new face to database")
        print("2. Compare a face to database")
        print("3. Exit")

        choice = input("Select option: ")

        if choice == '1':
            img_path = input("Image path: ")
            name = input("Name to save as: ")
            add_face(img_path, name)
        elif choice == '2':
            img_path = input("Image path: ")
            compare_face(img_path)
        elif choice == '3':
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    main()
