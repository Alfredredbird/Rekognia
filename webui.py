import os
import pickle
from flask import Flask, request, render_template_string, redirect, url_for
import face_recognition
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
DB_IMAGE_FOLDER = 'static/db_faces'
KNOWN_FACES_FILE = 'known_faces.pkl'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_IMAGE_FOLDER, exist_ok=True)

# Load known faces
if os.path.exists(KNOWN_FACES_FILE):
    with open(KNOWN_FACES_FILE, 'rb') as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

# ---------- Helper functions ----------
def save_known_faces():
    with open(KNOWN_FACES_FILE, 'wb') as f:
        pickle.dump(known_faces, f)

def update_training(name, new_encoding):
    """Averaging new encoding into existing one."""
    if name in known_faces:
        known_faces[name] = (known_faces[name] + new_encoding) / 2
    else:
        known_faces[name] = new_encoding
    save_known_faces()

def match_face(encoding):
    for name, known_encoding in known_faces.items():
        match = face_recognition.compare_faces([known_encoding], encoding, tolerance=0.45)[0]
        if match:
            # Return name and corresponding file path
            filename = secure_filename(name + '.jpg')
            return name, os.path.join(DB_IMAGE_FOLDER, filename)
    return None, None


# ---------- Routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    message = ''
    result_name = ''
    uploaded_image_url = ''
    matched_image_url = None 
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            message = 'No file selected.'
        else:
            filename = secure_filename(file.filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                message = 'No face detected in image.'
            else:
                encoding = encodings[0]
                matched_name = match_face(encoding)

                matched_name, matched_path = match_face(encoding)

                if matched_name:
                    result_name = matched_name
                    update_training(matched_name, encoding)
                    uploaded_image_url = '/' + image_path
                    matched_image_url = '/' + matched_path
                    message = f'✅ Matched with: {matched_name}'
                else:
                    message = '❌ No match found.'
                    uploaded_image_url = '/' + image_path
                    matched_image_url = None
                    return render_template_string(TEMPLATE_ADD_NEW, image_url=uploaded_image_url, encoding=encoding.tolist())


    return render_template_string(
    TEMPLATE_INDEX,
    message=message,
    result=result_name,
    image_url=uploaded_image_url,
    matched_image_url=matched_image_url,
    db_faces=os.listdir(DB_IMAGE_FOLDER)
)



@app.route('/add', methods=['POST'])
def add_face():
    name = request.form['name'].strip()
    encoding = np.array(eval(request.form['encoding']))  # string -> list -> np.array
    file_url = request.form['image_url'].lstrip('/')

    if name == '':
        return 'Name is required.', 400

    final_path = os.path.join(DB_IMAGE_FOLDER, secure_filename(name + '.jpg'))
    os.rename(file_url, final_path)

    update_training(name, encoding)
    return redirect(url_for('index'))


# ---------- HTML Templates ----------
TEMPLATE_INDEX = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Face Recognition</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap" rel="stylesheet">
  <style>
    body {
      background: #fef6ff;
      font-family: 'Poppins', sans-serif;
      color: #444;
      padding: 40px;
      text-align: center;
    }
    h1 {
      color: #9c27b0;
    }
    form {
      background: #ffffff;
      padding: 20px;
      border-radius: 16px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.05);
      display: inline-block;
      margin-bottom: 20px;
    }
    input[type=file] {
      padding: 10px;
      border: 2px solid #e1bee7;
      border-radius: 10px;
      margin: 10px;
      outline: none;
    }
    input[type=submit] {
      background: #ce93d8;
      color: white;
      border: none;
      padding: 10px 20px;
      font-weight: bold;
      border-radius: 10px;
      cursor: pointer;
      transition: background 0.3s ease;
    }
    input[type=submit]:hover {
      background: #ba68c8;
    }
    .message {
      margin: 20px auto;
      font-weight: bold;
      color: #6a1b9a;
    }
    .image-grid {
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 40px;
      margin-top: 30px;
    }
    .image-preview {
      border-radius: 16px;
      box-shadow: 0 4px 15px rgba(0,0,0,0.1);
      max-width: 300px;
    }
    .image-caption {
      margin-top: 10px;
      font-size: 14px;
      color: #888;
    }
  </style>
</head>
<body>
  <h1>Facial Recognition</h1>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="image">
    <input type="submit" value="Upload and Check">
  </form>

  <div class="message">{{ message }}</div>

  {% if image_url %}
    <div class="image-grid">
      <div>
        <img class="image-preview" src="{{ image_url }}">
        <div class="image-caption">Your Upload</div>
      </div>
      {% if matched_image_url %}
      <div>
        <img class="image-preview" src="{{ matched_image_url }}">
        <div class="image-caption">Best Match: {{ result }}</div>
      </div>
      {% endif %}
    </div>
  {% endif %}
</body>
</html>
'''



TEMPLATE_ADD_NEW = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Add New Face</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap" rel="stylesheet">
  <style>
    body {
      background: #f3f7fa;
      font-family: 'Poppins', sans-serif;
      color: #333;
      padding: 40px;
      text-align: center;
    }
    h1 {
      color: #4a148c;
    }
    form {
      background: #fff;
      padding: 20px;
      border-radius: 14px;
      box-shadow: 0 5px 20px rgba(0,0,0,0.07);
      display: inline-block;
    }
    input[type=text] {
      padding: 10px;
      border-radius: 10px;
      border: 2px solid #ce93d8;
      margin: 10px;
      outline: none;
      width: 250px;
    }
    input[type=submit] {
      background: #81d4fa;
      color: #000;
      border: none;
      padding: 10px 25px;
      font-weight: bold;
      border-radius: 10px;
      cursor: pointer;
      transition: background 0.3s ease;
    }
    input[type=submit]:hover {
      background: #4fc3f7;
    }
    img {
      border-radius: 12px;
      margin: 20px auto;
      box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
  </style>
</head>
<body>
  <h1>New Face Detected</h1>
  <img src="{{ image_url }}" width="300"><br><br>
  <form method="post" action="/add">
    <input type="hidden" name="encoding" value="{{ encoding }}">
    <input type="hidden" name="image_url" value="{{ image_url }}">
    <label>
      <input type="text" name="name" placeholder="Enter name" required>
    </label>
    <br>
    <input type="submit" value="Add to Database">
  </form>
</body>
</html>
'''


# ---------- Run ----------
if __name__ == '__main__':
    app.run(debug=True)
