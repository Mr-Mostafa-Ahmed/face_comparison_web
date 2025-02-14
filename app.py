import time
import os
import insightface
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from insightface.app import FaceAnalysis
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # Reduced to 8MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize face analysis with lighter configuration
face_analyzer = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
face_analyzer.prepare(ctx_id=0, det_size=(320, 320))  # Reduced detection size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}  # Removed GIF support

def compare_faces(image1_path, image2_path):
    try:
        # Read and resize images to reduce memory usage
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
        if img1 is None or img2 is None:
            return {
                'success': False,
                'error': "Could not read one or both images"
            }
        
        # Resize large images to reduce memory usage
        max_size = 640
        if img1.shape[0] > max_size or img1.shape[1] > max_size:
            scale = max_size / max(img1.shape[0], img1.shape[1])
            img1 = cv2.resize(img1, None, fx=scale, fy=scale)
        if img2.shape[0] > max_size or img2.shape[1] > max_size:
            scale = max_size / max(img2.shape[0], img2.shape[1])
            img2 = cv2.resize(img2, None, fx=scale, fy=scale)
            
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
        
        # Determine results with more detailed levels
        if similarity_score >= 90:
            confidence = "very high"
            is_same = True
            message = "These images are definitely of the same person."
            color = "green"
        elif similarity_score >= 80:
            confidence = "high"
            is_same = True
            message = "These images appear to be the same person."
            color = "green"
        elif similarity_score >= 70:
            confidence = "moderately high"
            is_same = True
            message = "These images are likely the same person."
            color = "green"
        elif similarity_score >= 60:
            confidence = "moderate"
            is_same = True
            message = "These images might be the same person."
            color = "yellow"
        else:
            confidence = "low"
            is_same = False
            message = "These images appear to be different people."
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
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({
                'success': False, 
                'error': "Please upload both images"
            })
            
        image1 = request.files['image1']
        image2 = request.files['image2']
        
        if image1.filename == '' or image2.filename == '':
            return jsonify({'success': False, 'error': 'No selected files'})
            
        if not (image1 and allowed_file(image1.filename) and image2 and allowed_file(image2.filename)):
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload JPG or PNG images'})
            
        timestamp = str(int(time.time() * 1000))
        image1_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_1_{secure_filename(image1.filename)}")
        image2_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_2_{secure_filename(image2.filename)}")
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        image1.save(image1_path)
        image2.save(image2_path)
        
        result = compare_faces(image1_path, image2_path)
        
        try:
            os.remove(image1_path)
            os.remove(image2_path)
        except Exception as e:
            print(f"Error cleaning up files: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)