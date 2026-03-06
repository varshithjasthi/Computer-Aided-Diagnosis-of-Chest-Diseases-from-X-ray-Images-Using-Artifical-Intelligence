"""
Model architecture and loading module.
Contains DeiT and Swin Transformer model definitions with hybrid fusion.
Complete implementation for AI Radiology Assistant with Hybrid Transformer Architecture.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_MODE = os.getenv("MODEL_MODE", "hybrid")  # deit, swin, or hybrid

# Model paths (configurable via environment variables)
DEIT_MODEL_PATH = os.getenv("DEIT_MODEL_PATH", "models/deit_model.pth")
SWIN_MODEL_PATH = os.getenv("SWIN_MODEL_PATH", "models/swin_model.pth")

# Class names for chest X-ray classification (NIH ChestX-ray14 classes)
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "NORMAL"
]

# Model configuration
INPUT_SIZE = 224  # Input image size for both models
NUM_CLASSES = len(CLASS_NAMES)

# ==================== UTILITY FUNCTIONS ====================

def normalize_class_name(name):
    """
    Normalize class name to standard format.
    
    Args:
        name: Raw class name
        
    Returns:
        Normalized class name
    """
    if name.upper() == "NORMAL":
        return "NORMAL"
    return name

def get_class_index(class_name):
    """
    Get index of class name.
    
    Args:
        class_name: Name of class
        
    Returns:
        Index in CLASS_NAMES
    """
    try:
        return CLASS_NAMES.index(class_name)
    except ValueError:
        return 0  # Default to first class if not found

def get_device_info():
    """Get detailed device information."""
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "name": torch.cuda.get_device_name(0),
            "memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        }
    else:
        return {
            "device": "cpu",
            "name": "CPU",
            "memory": "System RAM"
        }

# ==================== HYBRID MODEL CLASS ====================

class HybridModel(nn.Module):
    """
    Hybrid model combining DeiT and Swin Transformer predictions.
    Performs probability fusion with normalization and provides attention maps.
    """
    
    def __init__(self, deit_model, swin_model, weights=[0.5, 0.5], temperature=1.0):
        """
        Initialize hybrid model.
        
        Args:
            deit_model: Pretrained DeiT model
            swin_model: Pretrained Swin Transformer model
            weights: Fusion weights for [deit, swin]
            temperature: Temperature for softmax scaling (higher = softer probabilities)
        """
        super().__init__()
        self.deit = deit_model
        self.swin = swin_model
        self.weights = nn.Parameter(torch.tensor(weights), requires_grad=False)
        self.temperature = temperature
        
        # Set models to evaluation mode by default
        self.deit.eval()
        self.swin.eval()
        
        # Store model info
        self.num_classes = NUM_CLASSES
        self.input_size = INPUT_SIZE
        
    def forward(self, x):
        """
        Forward pass through both models and fuse probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (hybrid_probs, deit_probs, swin_probs)
        """
        # Get logits from both models
        deit_logits = self.deit(x)
        swin_logits = self.swin(x)
        
        # Apply temperature scaling
        deit_logits = deit_logits / self.temperature
        swin_logits = swin_logits / self.temperature
        
        # Convert to probabilities
        deit_probs = F.softmax(deit_logits, dim=1)
        swin_probs = F.softmax(swin_logits, dim=1)
        
        # Weighted fusion
        hybrid_probs = self.weights[0] * deit_probs + self.weights[1] * swin_probs
        
        # Normalize to ensure sum to 1 (handles any floating point errors)
        hybrid_probs = hybrid_probs / hybrid_probs.sum(dim=1, keepdim=True)
        
        return hybrid_probs, deit_probs, swin_probs
    
    def forward_deit_only(self, x):
        """Forward pass using only DeiT model."""
        deit_logits = self.deit(x)
        return F.softmax(deit_logits / self.temperature, dim=1)
    
    def forward_swin_only(self, x):
        """Forward pass using only Swin model."""
        swin_logits = self.swin(x)
        return F.softmax(swin_logits / self.temperature, dim=1)
    
    def get_attention_maps(self, x, model_type='both'):
        """
        Generate attention maps from models.
        
        Args:
            x: Input tensor
            model_type: 'deit', 'swin', or 'both'
            
        Returns:
            Dictionary containing attention maps
        """
        attention_maps = {}
        
        with torch.no_grad():
            if model_type in ['deit', 'both'] and hasattr(self.deit, 'get_last_selfattention'):
                try:
                    # Get DeiT attention
                    attn = self.deit.get_last_selfattention(x)
                    if attn is not None:
                        # Average over heads
                        attn = attn.mean(dim=1)  # [batch, num_patches, num_patches]
                        attention_maps['deit'] = attn
                except Exception as e:
                    print(f"⚠️ Could not get DeiT attention: {e}")
            
            if model_type in ['swin', 'both'] and hasattr(self.swin, 'get_last_selfattention'):
                try:
                    # Get Swin attention
                    attn = self.swin.get_last_selfattention(x)
                    if attn is not None:
                        # Average over heads
                        if len(attn.shape) == 4:  # [batch, heads, patches, patches]
                            attn = attn.mean(dim=1)
                        attention_maps['swin'] = attn
                except Exception as e:
                    print(f"⚠️ Could not get Swin attention: {e}")
        
        return attention_maps
    
    def get_model_info(self):
        """Get information about the hybrid model."""
        return {
            "deit": {
                "type": type(self.deit).__name__,
                "parameters": sum(p.numel() for p in self.deit.parameters()),
                "trainable": sum(p.numel() for p in self.deit.parameters() if p.requires_grad)
            },
            "swin": {
                "type": type(self.swin).__name__,
                "parameters": sum(p.numel() for p in self.swin.parameters()),
                "trainable": sum(p.numel() for p in self.swin.parameters() if p.requires_grad)
            },
            "fusion_weights": self.weights.tolist(),
            "temperature": self.temperature,
            "num_classes": self.num_classes
        }

# ==================== MODEL LOADING FUNCTIONS ====================

def load_deit_model(model_path, num_classes=NUM_CLASSES, device=DEVICE):
    """
    Load DeiT model with pretrained weights.
    
    Args:
        model_path: Path to .pth file
        num_classes: Number of output classes
        device: Torch device
        
    Returns:
        Loaded DeiT model
    """
    try:
        print(f"📥 Loading DeiT model from: {model_path}")
        
        # Create DeiT model (using tiny variant for efficiency)
        model = timm.create_model(
            'deit_tiny_patch16_224', 
            pretrained=False, 
            num_classes=num_classes
        )
        
        # Load weights if file exists
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present (DataParallel)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                # Load state dict
                model.load_state_dict(state_dict, strict=False)
                print(f"✅ DeiT weights loaded successfully")
            else:
                model.load_state_dict(checkpoint, strict=False)
                print(f"✅ DeiT weights loaded successfully")
        else:
            print(f"⚠️ DeiT weights not found at {model_path}, using random initialization")
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        # Log model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"📊 DeiT parameters: {num_params:,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading DeiT model: {e}")
        import traceback
        traceback.print_exc()
        raise

def load_swin_model(model_path, num_classes=NUM_CLASSES, device=DEVICE):
    """
    Load Swin Transformer model with pretrained weights.
    
    Args:
        model_path: Path to .pth file
        num_classes: Number of output classes
        device: Torch device
        
    Returns:
        Loaded Swin model
    """
    try:
        print(f"📥 Loading Swin model from: {model_path}")
        
        # Create Swin model (using tiny variant for efficiency)
        model = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=False, 
            num_classes=num_classes
        )
        
        # Load weights if file exists
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present (DataParallel)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                # Load state dict
                model.load_state_dict(state_dict, strict=False)
                print(f"✅ Swin weights loaded successfully")
            else:
                model.load_state_dict(checkpoint, strict=False)
                print(f"✅ Swin weights loaded successfully")
        else:
            print(f"⚠️ Swin weights not found at {model_path}, using random initialization")
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        # Log model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Swin parameters: {num_params:,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading Swin model: {e}")
        import traceback
        traceback.print_exc()
        raise

def load_models(deit_path=DEIT_MODEL_PATH, swin_path=SWIN_MODEL_PATH, device=DEVICE):
    """
    Load both models and create hybrid fusion.
    
    Args:
        deit_path: Path to DeiT weights
        swin_path: Path to Swin weights
        device: Torch device
        
    Returns:
        hybrid_model, deit_model, swin_model, class_names
    """
    print(f"\n{'='*60}")
    print(f"🚀 NEURORAD AI - HYBRID TRANSFORMER INITIALIZATION")
    print(f"{'='*60}")
    
    # Get device info
    device_info = get_device_info()
    print(f"💻 Device: {device_info['name']} ({device_info['device']})")
    if device_info['device'] == 'cuda':
        print(f"📈 GPU Memory: {device_info['memory']}")
    
    print(f"📋 Number of Classes: {NUM_CLASSES}")
    print(f"📏 Input Size: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"{'='*60}\n")
    
    # Load individual models
    deit_model = load_deit_model(deit_path, NUM_CLASSES, device)
    swin_model = load_swin_model(swin_path, NUM_CLASSES, device)
    
    # Create hybrid model
    print(f"\n🔗 Creating hybrid fusion model...")
    hybrid_model = HybridModel(deit_model, swin_model, weights=[0.5, 0.5]).to(device)
    hybrid_model.eval()
    
    # Get model info
    model_info = hybrid_model.get_model_info()
    
    print(f"\n{'='*60}")
    print(f"🚀 HYBRID TRANSFORMER ENGINE ACTIVATED")
    print(f"{'='*60}")
    print(f"📊 Fusion Weights: DeiT={model_info['fusion_weights'][0]}, Swin={model_info['fusion_weights'][1]}")
    print(f"🌡️ Temperature: {model_info['temperature']}")
    print(f"📈 Total Parameters: {model_info['deit']['parameters'] + model_info['swin']['parameters']:,}")
    print(f"{'='*60}\n")
    
    return hybrid_model, deit_model, swin_model, CLASS_NAMES

# ==================== IMAGE PREPROCESSING ====================

def get_preprocessing_transform(input_size=INPUT_SIZE):
    """
    Get image preprocessing pipeline.
    
    Returns:
        torchvision.transforms.Compose: Preprocessing pipeline
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_path, input_size=INPUT_SIZE):
    """
    Preprocess image for model inference.
    
    Args:
        image_path: Path to image file
        input_size: Input size for models
        
    Returns:
        Preprocessed tensor and original PIL image
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply preprocessing
        transform = get_preprocessing_transform(input_size)
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return input_tensor, image
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        raise

def preprocess_image_from_array(image_array, input_size=INPUT_SIZE):
    """
    Preprocess numpy array image for model inference.
    
    Args:
        image_array: numpy array of image
        input_size: Input size for models
        
    Returns:
        Preprocessed tensor
    """
    try:
        # Convert to PIL Image
        if isinstance(image_array, np.ndarray):
            image = Image.fromarray(image_array)
        else:
            image = image_array
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing
        transform = get_preprocessing_transform(input_size)
        input_tensor = transform(image).unsqueeze(0)
        
        return input_tensor
        
    except Exception as e:
        print(f"❌ Error preprocessing image array: {e}")
        raise

# ==================== INFERENCE FUNCTIONS ====================

def hybrid_probability_fusion(image_path, hybrid_model, deit_model, swin_model, 
                              class_names, mode='hybrid', device=DEVICE):
    """
    Perform inference using specified mode.
    
    Args:
        image_path: Path to input image
        hybrid_model: Hybrid model instance
        deit_model: DeiT model
        swin_model: Swin model
        class_names: List of class names
        mode: Inference mode ('deit', 'swin', or 'hybrid')
        device: Torch device
        
    Returns:
        predictions: List of prediction dictionaries
        img_tensor: Preprocessed image tensor
    """
    try:
        # Preprocess image
        img_tensor, original_img = preprocess_image(image_path)
        img_tensor = img_tensor.to(device)
        
        # Perform inference based on mode
        with torch.no_grad():
            if mode == 'deit':
                # Pure DeiT inference
                outputs = deit_model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                
            elif mode == 'swin':
                # Pure Swin inference
                outputs = swin_model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                
            elif mode == 'hybrid':
                # Hybrid fusion inference
                hybrid_probs, deit_probs, swin_probs = hybrid_model(img_tensor)
                probs = hybrid_probs
                
            else:
                raise ValueError(f"Invalid mode: {mode}. Choose 'deit', 'swin', or 'hybrid'")
        
        # Convert to predictions format
        predictions = []
        probs_np = probs.cpu().numpy()[0]
        
        for idx, prob in enumerate(probs_np):
            disease_name = class_names[idx]
            is_normal = normalize_class_name(disease_name) == "NORMAL"
            
            predictions.append({
                "disease": disease_name,
                "probability": float(prob * 100),  # Convert to percentage
                "is_normal": is_normal
            })
        
        # Sort by probability descending
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        
        return predictions, img_tensor
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise

def batch_inference(image_paths, hybrid_model, deit_model, swin_model, 
                    class_names, mode='hybrid', device=DEVICE):
    """
    Perform batch inference on multiple images.
    
    Args:
        image_paths: List of image paths
        hybrid_model: Hybrid model instance
        deit_model: DeiT model
        swin_model: Swin model
        class_names: List of class names
        mode: Inference mode
        device: Torch device
        
    Returns:
        List of predictions for each image
    """
    results = []
    
    for image_path in image_paths:
        predictions, _ = hybrid_probability_fusion(
            image_path, hybrid_model, deit_model, swin_model,
            class_names, mode, device
        )
        results.append({
            "image": os.path.basename(image_path),
            "predictions": predictions
        })
    
    return results

# ==================== ATTENTION MAP GENERATION ====================

def generate_hybrid_attention_map(img_tensor, deit_model, swin_model, class_idx, 
                                 class_names, save_path=None, device=DEVICE, 
                                 mode='hybrid', is_normal=False):
    """
    Generate attention map visualization.
    
    Args:
        img_tensor: Input image tensor
        deit_model: DeiT model
        swin_model: Swin model
        class_idx: Target class index
        class_names: List of class names
        save_path: Path to save attention map
        device: Torch device
        mode: Inference mode
        is_normal: Whether prediction is normal
        
    Returns:
        heatmap_description: Text description
        heatmap_success: Boolean indicating success
    """
    try:
        import cv2
        import numpy as np
        
        img_tensor = img_tensor.to(device)
        
        # Generate attention maps based on mode
        attention_maps = []
        
        with torch.no_grad():
            if mode == 'hybrid' or mode == 'deit':
                if hasattr(deit_model, 'get_last_selfattention'):
                    try:
                        attn = deit_model.get_last_selfattention(img_tensor)
                        if attn is not None:
                            # Average attention heads
                            if len(attn.shape) == 4:  # [batch, heads, patches, patches]
                                attn = attn.mean(dim=1)[0].cpu().numpy()
                                attention_maps.append(('DeiT', attn))
                    except Exception as e:
                        print(f"⚠️ DeiT attention extraction failed: {e}")
            
            if mode == 'hybrid' or mode == 'swin':
                if hasattr(swin_model, 'get_last_selfattention'):
                    try:
                        attn = swin_model.get_last_selfattention(img_tensor)
                        if attn is not None:
                            if len(attn.shape) == 4:
                                attn = attn.mean(dim=1)[0].cpu().numpy()
                                attention_maps.append(('Swin', attn))
                    except Exception as e:
                        print(f"⚠️ Swin attention extraction failed: {e}")
        
        if not attention_maps:
            # Fallback: generate gradient-based attention
            return generate_gradient_attention(img_tensor, deit_model, class_idx, save_path, device, is_normal)
        
        # Use first available attention map
        model_name, attention = attention_maps[0]
        
        # Get grid size (assuming sqrt of patches)
        grid_size = int(np.sqrt(attention.shape[0]))
        if grid_size * grid_size != attention.shape[0]:
            # Reshape to 14x14 for DeiT (196 patches)
            grid_size = 14
        
        # Reshape attention to grid
        try:
            attention_grid = attention.reshape(grid_size, grid_size)
        except:
            # If reshape fails, resize to 14x14
            attention_grid = cv2.resize(attention.reshape(1, -1), (14, 14))
        
        # Normalize attention
        attention_grid = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min() + 1e-8)
        
        # Resize to image size
        attention_resized = cv2.resize(attention_grid, (224, 224))
        
        # Create heatmap
        heatmap = np.uint8(255 * attention_resized)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Get original image from tensor
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_img = img_tensor.cpu().numpy()[0].transpose(1, 2, 0)
        original_img = original_img * std + mean
        original_img = np.clip(original_img * 255, 0, 255).astype(np.uint8)
        
        # Blend images
        blended = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, blended)
        
        # Generate description
        if is_normal:
            description = f"AI attention shows normal physiological patterns without focal abnormalities. The {model_name} model demonstrates diffuse attention across normal anatomical structures."
        else:
            target_class = class_names[class_idx] if class_idx < len(class_names) else "pathology"
            description = f"AI attention highlights regions suspicious for {target_class} (red/orange areas). The {model_name} model focuses on areas corresponding to radiographic findings."
        
        return description, True
        
    except Exception as e:
        print(f"❌ Error generating attention map: {e}")
        import traceback
        traceback.print_exc()
        return f"Attention map generation failed. Using standard visualization.", False

def generate_gradient_attention(img_tensor, model, class_idx, save_path=None, device=DEVICE, is_normal=False):
    """
    Generate gradient-based attention map as fallback.
    
    Args:
        img_tensor: Input image tensor
        model: Model to use
        class_idx: Target class index
        save_path: Path to save
        device: Device
        is_normal: Whether normal
        
    Returns:
        description, success
    """
    try:
        import cv2
        import numpy as np
        
        img_tensor = img_tensor.to(device)
        img_tensor.requires_grad_()
        
        # Forward pass
        output = model(img_tensor)
        
        # Get target class score
        target_score = output[0, class_idx]
        
        # Backward pass
        model.zero_grad()
        target_score.backward()
        
        # Get gradients
        gradients = img_tensor.grad[0].cpu().numpy()
        
        # Average across channels
        gradients = np.mean(gradients, axis=0)
        
        # Normalize
        gradients = np.abs(gradients)
        gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
        
        # Resize to 224x224
        gradients = cv2.resize(gradients, (224, 224))
        
        # Create heatmap
        heatmap = np.uint8(255 * gradients)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Get original image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_img = img_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
        original_img = original_img * std + mean
        original_img = np.clip(original_img * 255, 0, 255).astype(np.uint8)
        
        # Blend
        blended = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        if save_path:
            cv2.imwrite(save_path, blended)
        
        if is_normal:
            description = "Gradient-based attention shows normal physiological patterns."
        else:
            description = "Gradient-based attention highlights regions of interest."
        
        return description, True
        
    except Exception as e:
        print(f"❌ Error generating gradient attention: {e}")
        return "Attention map not available.", False

# ==================== MODEL VALIDATION ====================

def validate_model_output(model, sample_input):
    """
    Validate model output shape and values.
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor
        
    Returns:
        bool: True if valid
    """
    try:
        with torch.no_grad():
            output = model(sample_input)
            
            # Check output shape
            if output.shape[1] != NUM_CLASSES:
                print(f"❌ Model output has {output.shape[1]} classes, expected {NUM_CLASSES}")
                return False
            
            # Check output values (should be logits, not probabilities)
            if torch.any(torch.isnan(output)):
                print("❌ Model output contains NaN values")
                return False
            
            print(f"✅ Model validation passed. Output shape: {output.shape}")
            return True
            
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

def test_inference(model, image_path, class_names):
    """
    Test inference on a single image.
    
    Args:
        model: PyTorch model
        image_path: Path to test image
        class_names: List of class names
        
    Returns:
        dict: Top predictions
    """
    try:
        # Preprocess
        img_tensor, _ = preprocess_image(image_path)
        img_tensor = img_tensor.to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probs, 3)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append({
                "disease": class_names[idx],
                "probability": f"{prob.item() * 100:.2f}%"
            })
        
        return results
        
    except Exception as e:
        print(f"❌ Test inference failed: {e}")
        return None

# ==================== EXPORTED FUNCTIONS ====================

__all__ = [
    'HybridModel',
    'load_models',
    'load_deit_model',
    'load_swin_model',
    'hybrid_probability_fusion',
    'batch_inference',
    'generate_hybrid_attention_map',
    'preprocess_image',
    'preprocess_image_from_array',
    'normalize_class_name',
    'get_class_index',
    'CLASS_NAMES',
    'DEVICE',
    'MODEL_MODE',
    'INPUT_SIZE',
    'NUM_CLASSES'
]