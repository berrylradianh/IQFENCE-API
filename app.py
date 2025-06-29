from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import face_recognition
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import firebase_admin
from firebase_admin import credentials, firestore
import io
import os
import tempfile
from datetime import datetime
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Firebase with environment variable
firebase_cred_path = os.getenv('FIREBASE_CRED_PATH')
cred = credentials.Certificate(firebase_cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.file']
def get_drive_service():
    creds = None
    token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
    creds_path = os.getenv('GOOGLE_CREDS_PATH', 'credentials.json')
    
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def upload_to_drive(file_data, filename):
    drive_service = get_drive_service()
    folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
    file_metadata = {
        'name': filename,
        'parents': [folder_id]
    }
    
    # Create a temporary file to store the image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(file_data)
        temp_file_path = temp_file.name
    
    # Upload the file using the temporary file path
    media = MediaFileUpload(temp_file_path, mimetype='image/jpeg')
    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webViewLink'
    ).execute()
    
    # Delete the temporary file
    os.unlink(temp_file_path)
    
    return file.get('webViewLink')

# Load Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/upload', methods=['POST'])
def upload_photo():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    karyawan_id = request.form.get('karyawan_id')
    
    # Read image
    filestr = file.read()
    nparr = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return jsonify({'error': 'No face detected. Please upload another photo.'}), 400
    
    # Upload to Google Drive
    filename = f"karyawan_{karyawan_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    photo_url = upload_to_drive(filestr, filename)
    
    # Update karyawan collection
    karyawan_ref = db.collection('karyawan').document(karyawan_id)
    karyawan_ref.update({
        'foto': photo_url
    })
    
    return jsonify({'message': 'Photo uploaded successfully', 'photo_url': photo_url}), 200

@app.route('/presensi', methods=['POST'])
def presensi():
    from flask import request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    karyawan_id = request.form.get('karyawan_id')
    
    # Read uploaded image
    filestr = file.read()
    nparr = np.frombuffer(filestr, np.uint8)
    uploaded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Face detection for uploaded image
    gray = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return jsonify({'error': 'No face detected. Please take another photo.'}), 400
    
    # Get stored photo from karyawan collection
    karyawan_ref = db.collection('karyawan').document(karyawan_id)
    karyawan = karyawan_ref.get()
    
    if not karyawan.exists:
        return jsonify({'error': 'Karyawan not found'}), 404
    
    stored_photo_url = karyawan.to_dict().get('foto')
    if not stored_photo_url:
        return jsonify({'error': 'No stored photo found for this karyawan'}), 404
    
    # Download stored photo from Google Drive
    drive_service = get_drive_service()
    file_id = stored_photo_url.split('/')[5].split('?')[0]
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    
    # Load stored image
    stored_nparr = np.frombuffer(fh.getvalue(), np.uint8)
    stored_img = cv2.imdecode(stored_nparr, cv2.IMREAD_COLOR)
    if stored_img is None:
        return jsonify({'error': 'Failed to load stored image from Google Drive'}), 500
    
    # Convert images for face_recognition
    uploaded_rgb = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2RGB)
    stored_rgb = cv2.cvtColor(stored_img, cv2.COLOR_BGR2RGB)
    
    # Get face encodings
    uploaded_encodings = face_recognition.face_encodings(uploaded_rgb)
    stored_encodings = face_recognition.face_encodings(stored_rgb)
    
    if not uploaded_encodings or not stored_encodings:
        return jsonify({'error': 'Unable to process face encodings. Please try again.'}), 400
    
    # Compare faces
    matches = face_recognition.compare_faces([stored_encodings[0]], uploaded_encodings[0])
    
    if matches[0]:
        return jsonify({'message': 'Face matched successfully'}), 200
    else:
        return jsonify({'error': 'Face does not match. Please take another photo.'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)