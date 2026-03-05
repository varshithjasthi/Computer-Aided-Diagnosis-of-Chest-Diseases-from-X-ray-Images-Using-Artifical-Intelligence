# src/app.py

import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
import traceback
import sys

# Add parent directory to path for imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Disease classes
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "Normal"
]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model loading
class XRayClassifier:
    def __init__(self, model_path='saved_model/xray_15class_model.pt'):
        self.model = None
        self.model_path = model_path
        self.device = device
        self.transform = self._get_transforms()
        self.load_model()
    
    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        try:
            # Load the model
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model architecture (adjust based on your model)
            # This is a placeholder - replace with your actual model architecture
            # from models.swin_transformer import SwinTransformer
            # from models.dit import DiT
            
            # For demonstration, using a simple model structure
            # Replace this with your actual model initialization
            class SimpleModel(nn.Module):
                def __init__(self, num_classes=15):
                    super().__init__()
                    self.fc = nn.Linear(224*224*3, num_classes)
                
                def forward(self, x):
                    x = x.view(x.size(0), -1)
                    return self.fc(x)
            
            self.model = SimpleModel(num_classes=15)
            
            # Load state dict
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
            self.model = None
    
    def preprocess_image(self, image_path):
        """Preprocess image for model inference"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            return image_tensor.to(self.device)
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")
    
    @torch.no_grad()
    def predict(self, image_tensor):
        """Run inference and return predictions"""
        if self.model is None:
            raise Exception("Model not loaded")
        
        outputs = self.model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, 3)
        
        results = []
        for i in range(3):
            class_idx = top_indices[0][i].item()
            confidence = top_probs[0][i].item()
            results.append({
                'class': CLASSES[class_idx],
                'confidence': round(confidence * 100, 2),
                'index': class_idx
            })
        
        return results

# Initialize classifier
model_path = os.path.join('saved_model', 'xray_15class_model.pt')
classifier = XRayClassifier(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file exists in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload PNG, JPG, or JPEG files.'}), 400
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess image
            image_tensor = classifier.preprocess_image(filepath)
            
            # Make prediction
            predictions = classifier.predict(image_tensor)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            # Return predictions
            return jsonify({
                'success': True,
                'predictions': predictions,
                'primary_diagnosis': predictions[0]['class'],
                'confidence': predictions[0]['confidence']
            })
            
        except Exception as e:
            # Clean up file if error occurs
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)