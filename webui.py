import os
import face_recognition
import numpy as np
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'secret'
UPLOAD_FOLDER = 'static/uploads'
DB_FOLDER = 'database'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

def load_database():
    database = {}
    for file in os.listdir(DB_FOLDER):
        if file.endswith('.pkl'):
            with open(os.path.join(DB_FOLDER, file), 'rb') as f:
                database[file[:-4]] = pickle.load(f)
    return database

def save_face(name, encoding):
    with open(os.path.join(DB_FOLDER, f"{name}.pkl"), 'wb') as f:
        pickle.dump(encoding, f)

def detect_face(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None

def update_training(name, new_encoding, alpha=0.3):
    # Blend old encoding with new for incremental training
    db = load_database()
    if name in db:
        updated = (1 - alpha) * db[name] + alpha * new_encoding
        save_face(name, updated)
        return True
    return False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        action = request.form.get('action')
        name = request.form.get('name', '').strip()
        file = request.files['image']

        if not file or file.filename == '':
            flash('No image selected.')
            return redirect(url_for('index'))

        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        encoding = detect_face(path)
        if encoding is None:
            flash("No face detected.")
            return redirect(url_for('index'))

        if action == 'add' and name:
            save_face(name, encoding)
            flash(f"Added {name} to database.")
        elif action == 'compare':
            db = load_database()
            if not db:
                flash("No faces in the database.")
                return redirect(url_for('index'))

            known_names = list(db.keys())
            known_encodings = list(db.values())

            matches = face_recognition.compare_faces(known_encodings, encoding)
            distances = face_recognition.face_distance(known_encodings, encoding)

            if True in matches:
                idx = matches.index(True)
                matched_name = known_names[idx]
                update_training(matched_name, encoding)
                flash(f"✅ Match found: {matched_name} (auto-trained)")
            else:
                idx = np.argmin(distances)
                flash(f"🤏 Closest match: {known_names[idx]} (distance: {distances[idx]:.2f})")
        else:
            flash("Invalid action or missing name.")
        return redirect(url_for('index'))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
