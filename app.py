from flask import Flask, render_template, request, jsonify
import time
import os
import insightface
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize face analysis
face_analyzer = FaceAnalysis(providers=['CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def compare_faces(image1_path, image2_path):
    try:
        # Read images
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
        if img1 is None or img2 is None:
            return {
                'success': False,
                'error': "Could not read one or both images"
            }
            
        # Convert BGR to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Get face embeddings
        faces1 = face_analyzer.get(img1)
        faces2 = face_analyzer.get(img2)
        
        if len(faces1) == 0 or len(faces2) == 0:
            return {
                'success': False,
                'error': "No faces detected in one or both images"
            }
            
        if len(faces1) > 1 or len(faces2) > 1:
            return {
                'success': False,
                'error': "Multiple faces detected. Please use images with single faces"
            }
            
        # Calculate similarity
        embedding1 = faces1[0].embedding
        embedding2 = faces2[0].embedding
        
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = float(similarity) * 100
        
        # Determine results
        if similarity_score >= 60:
            if similarity_score >= 80:
                confidence = "high"
                is_same = True
                message = "Same person with high confidence"
                color = "green"
            else:
                confidence = "medium"
                is_same = True
                message = "Likely the same person but with medium confidence"
                color = "orange"
        else:
            confidence = "high" if similarity_score < 40 else "medium"
            is_same = False
            message = "Different persons" if similarity_score < 40 else "Likely different persons"
            color = "red"
            
        return {
            'success': True,
            'similarity_score': round(similarity_score, 2),
            'is_same_person': is_same,
            'confidence': confidence,
            'message': message,
            'color': color
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error processing images: {str(e)}"
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    print("Files in request:", request.files)
    print("Request method:", request.method)
    
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({
            'success': False, 
            'error': f"Missing files. Found keys: {list(request.files.keys())}"
        })
        
    image1 = request.files['image1']
    image2 = request.files['image2']
    
    if image1.filename == '' or image2.filename == '':
        return jsonify({'success': False, 'error': 'No selected files'})
        
    if not (image1 and allowed_file(image1.filename) and image2 and allowed_file(image2.filename)):
        return jsonify({'success': False, 'error': 'Invalid file type. Please upload images (PNG, JPG, JPEG, GIF)'})
        
    try:
        # Save uploaded files
        image1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image1.filename))
        image2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image2.filename))
        
        image1.save(image1_path)
        image2.save(image2_path)
        
        # Compare faces
        result = compare_faces(image1_path, image2_path)
        
        # Clean up uploaded files
        os.remove(image1_path)
        os.remove(image2_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)