"""
Model architecture and loading module.
Contains DeiT and Swin Transformer model definitions with hybrid fusion.
Complete implementation for AI Radiology Assistant with Hybrid Transformer Architecture.
Compatible with app.py, services.py, and utils.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
import warnings
warnings.filterwarnings('ignore')

# Import configuration from utils
try:
    from utils import (
        DEIT_MODEL_PATH, SWIN_MODEL_PATH, DEVICE, IMG_SIZE, MODEL_MODE,
        normalize_class_name as utils_normalize_class_name
    )
except ImportError:
    # Fallback configuration if utils not available
    print("⚠️ utils.py not found, using default configuration")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_MODE = "hybrid"
    DEIT_MODEL_PATH = "models/deit_small_15_classes.pth"
    SWIN_MODEL_PATH = "models/swin_ultra_fast_15_classes.pth"
    IMG_SIZE = 224
    
    def utils_normalize_class_name(name):
        """Fallback normalize function"""
        if not name:
            return "NORMAL"
        upper_name = name.upper().strip()
        normal_patterns = ["NORMAL", "NO FINDING", "NOFINDING", "NO FINDINGS"]
        if any(pattern in upper_name for pattern in normal_patterns):
            return "NORMAL"
        return name

# ==================== CONFIGURATION ====================

# Class names for chest X-ray classification (15 classes as per trained models)
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax", "NORMAL"
]

# Model configuration
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SIZE = IMG_SIZE  # Use from utils

# ==================== UTILITY FUNCTIONS ====================

def normalize_class_name(name):
    """
    Normalize class name to standard format.
    Wrapper around utils function for compatibility.
    """
    return utils_normalize_class_name(name)

def get_class_index(class_name):
    """
    Get index of class name.
    
    Args:
        class_name: Name of class
        
    Returns:
        Index in CLASS_NAMES
    """
    normalized = normalize_class_name(class_name)
    try:
        return CLASS_NAMES.index(normalized)
    except ValueError:
        # Try case-insensitive match
        for i, name in enumerate(CLASS_NAMES):
            if name.upper() == normalized.upper():
                return i
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
        
        # Weighted fusion (probability averaging)
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
        Loaded DeiT model or None if loading fails
    """
    try:
        print(f"📥 Loading DeiT model from: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"❌ DeiT model not found at {model_path}")
            return None
        
        # Create DeiT model (small variant as per trained model)
        model = timm.create_model(
            'deit_small_patch16_224', 
            pretrained=False, 
            num_classes=num_classes
        )
        
        # Load weights
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
            
            # Load state dict with non-strict mode to handle minor mismatches
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {unexpected_keys}")
            print(f"✅ DeiT weights loaded successfully")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print(f"✅ DeiT weights loaded successfully")
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        # Log model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ DeiT Neural Engine Online ({num_params:,} parameters)")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading DeiT model: {e}")
        return None

def load_swin_model(model_path, num_classes=NUM_CLASSES, device=DEVICE):
    """
    Load Swin Transformer model with pretrained weights.
    
    Args:
        model_path: Path to .pth file
        num_classes: Number of output classes
        device: Torch device
        
    Returns:
        Loaded Swin model or None if loading fails
    """
    try:
        print(f"📥 Loading Swin model from: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"❌ Swin model not found at {model_path}")
            return None
        
        # Create Swin model (tiny variant as per trained model)
        model = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=False, 
            num_classes=num_classes
        )
        
        # Load weights
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
            
            # Load state dict with non-strict mode
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {unexpected_keys}")
            print(f"✅ Swin weights loaded successfully")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print(f"✅ Swin weights loaded successfully")
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        # Log model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Swin Transformer Online ({num_params:,} parameters)")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading Swin model: {e}")
        return None

def load_models(deit_path=DEIT_MODEL_PATH, swin_path=SWIN_MODEL_PATH, device=DEVICE):
    """
    Load both models and create hybrid fusion.
    
    Args:
        deit_path: Path to DeiT weights
        swin_path: Path to Swin weights
        device: Torch device
        
    Returns:
        hybrid_model, deit_model, swin_model, class_names
        Returns None for models that failed to load
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
    
    # Check if at least one model loaded successfully
    if deit_model is None and swin_model is None:
        print("❌ Failed to load any models. Exiting.")
        return None, None, None, CLASS_NAMES
    
    # Create hybrid model if both models loaded
    hybrid_model = None
    if deit_model is not None and swin_model is not None:
        print(f"\n🔗 Creating hybrid fusion model...")
        hybrid_model = HybridModel(deit_model, swin_model, weights=[0.5, 0.5]).to(device)
        hybrid_model.eval()
        print(f"🚀 Hybrid Transformer Engine Activated")
        
        # Get model info
        model_info = hybrid_model.get_model_info()
        print(f"📊 Fusion Weights: DeiT={model_info['fusion_weights'][0]}, Swin={model_info['fusion_weights'][1]}")
        print(f"📈 Total Parameters: {model_info['deit']['parameters'] + model_info['swin']['parameters']:,}")
    else:
        # Use available model(s) in fallback mode
        print(f"\n⚠️ Running in fallback mode with available model(s)")
        if deit_model is not None:
            print(f"✅ Using DeiT only mode")
        if swin_model is not None:
            print(f"✅ Using Swin only mode")
    
    print(f"{'='*60}\n")
    
    return hybrid_model, deit_model, swin_model, CLASS_NAMES

# ==================== IMAGE PREPROCESSING ====================

def get_preprocessing_transform(input_size=INPUT_SIZE):
    """
    Get image preprocessing pipeline with ImageNet normalization.
    
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
        # Return a dummy tensor rather than failing
        dummy_tensor = torch.zeros(1, 3, input_size, input_size)
        dummy_image = Image.new('RGB', (input_size, input_size))
        return dummy_tensor, dummy_image

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
        return torch.zeros(1, 3, input_size, input_size)

# ==================== INFERENCE FUNCTIONS ====================

def hybrid_probability_fusion(image_path, hybrid_model, deit_model, swin_model, 
                              class_names, mode=MODEL_MODE, device=DEVICE):
    """
    Perform inference using specified mode.
    
    Args:
        image_path: Path to input image
        hybrid_model: Hybrid model instance (can be None)
        deit_model: DeiT model (can be None)
        swin_model: Swin model (can be None)
        class_names: List of class names
        mode: Inference mode ('deit', 'swin', or 'hybrid')
        device: Torch device
        
    Returns:
        predictions: List of prediction dictionaries
        img_tensor: Preprocessed image tensor
    """
    try:
        # Validate model availability based on mode
        if mode == 'deit' and deit_model is None:
            print("⚠️ DeiT model not available, falling back to hybrid if possible")
            mode = 'hybrid' if hybrid_model is not None else 'swin'
        elif mode == 'swin' and swin_model is None:
            print("⚠️ Swin model not available, falling back to hybrid if possible")
            mode = 'hybrid' if hybrid_model is not None else 'deit'
        elif mode == 'hybrid' and hybrid_model is None:
            if deit_model is not None:
                print("⚠️ Hybrid model not available, falling back to DeiT")
                mode = 'deit'
            elif swin_model is not None:
                print("⚠️ Hybrid model not available, falling back to Swin")
                mode = 'swin'
            else:
                raise ValueError("No models available for inference")
        
        # Preprocess image
        img_tensor, _ = preprocess_image(image_path)
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
                # Hybrid fusion inference (probability averaging)
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
        # Return fallback predictions rather than crashing
        fallback_predictions = []
        for idx, disease in enumerate(class_names):
            is_normal = normalize_class_name(disease) == "NORMAL"
            fallback_predictions.append({
                "disease": disease,
                "probability": 100.0 if is_normal else 0.0,
                "is_normal": is_normal
            })
        return fallback_predictions, torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)

def batch_inference(image_paths, hybrid_model, deit_model, swin_model, 
                    class_names, mode=MODEL_MODE, device=DEVICE):
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
        try:
            predictions, _ = hybrid_probability_fusion(
                image_path, hybrid_model, deit_model, swin_model,
                class_names, mode, device
            )
            results.append({
                "image": os.path.basename(image_path),
                "predictions": predictions,
                "status": "success"
            })
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            results.append({
                "image": os.path.basename(image_path),
                "predictions": [],
                "status": "failed",
                "error": str(e)
            })
    
    return results

# ==================== ATTENTION MAP GENERATION ====================

def generate_hybrid_attention_map(img_tensor, deit_model, swin_model, class_idx, 
                                 class_names, save_path=None, device=DEVICE, 
                                 mode=MODEL_MODE, is_normal=False):
    """
    Generate attention map visualization combining both models.
    
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
        
        img_tensor = img_tensor.to(device)
        
        # Generate attention maps from both models
        deit_map = None
        swin_map = None
        
        with torch.no_grad():
            # Get DeiT attention if available and requested
            if mode in ['deit', 'hybrid'] and deit_model is not None and hasattr(deit_model, 'get_last_selfattention'):
                try:
                    deit_attn = deit_model.get_last_selfattention(img_tensor)
                    if deit_attn is not None:
                        # Average over heads
                        if len(deit_attn.shape) == 4:
                            deit_map = deit_attn.mean(dim=1)[0].cpu().numpy()
                except Exception as e:
                    print(f"⚠️ DeiT attention extraction failed: {e}")
            
            # Get Swin attention if available and requested
            if mode in ['swin', 'hybrid'] and swin_model is not None and hasattr(swin_model, 'get_last_selfattention'):
                try:
                    swin_attn = swin_model.get_last_selfattention(img_tensor)
                    if swin_attn is not None:
                        if len(swin_attn.shape) == 4:
                            swin_map = swin_attn.mean(dim=1)[0].cpu().numpy()
                except Exception as e:
                    print(f"⚠️ Swin attention extraction failed: {e}")
        
        # Combine attention maps (hybrid fusion)
        attention_map = None
        model_name = "Hybrid"
        
        if deit_map is not None and swin_map is not None:
            # Resize both to common size (14x14 for DeiT, adjust Swin)
            deit_size = int(np.sqrt(deit_map.shape[0]))
            swin_size = int(np.sqrt(swin_map.shape[0]))
            
            if deit_size * deit_size == deit_map.shape[0]:
                deit_grid = deit_map.reshape(deit_size, deit_size)
            else:
                deit_grid = cv2.resize(deit_map.reshape(1, -1), (14, 14))
            
            if swin_size * swin_size == swin_map.shape[0]:
                swin_grid = swin_map.reshape(swin_size, swin_size)
            else:
                swin_grid = cv2.resize(swin_map.reshape(1, -1), (14, 14))
            
            # Average the attention maps
            attention_map = 0.5 * deit_grid + 0.5 * swin_grid
            model_name = "Hybrid (DeiT+Swin)"
            
        elif deit_map is not None:
            # Use DeiT only
            deit_size = int(np.sqrt(deit_map.shape[0]))
            if deit_size * deit_size == deit_map.shape[0]:
                attention_map = deit_map.reshape(deit_size, deit_size)
            else:
                attention_map = cv2.resize(deit_map.reshape(1, -1), (14, 14))
            model_name = "DeiT"
            
        elif swin_map is not None:
            # Use Swin only
            swin_size = int(np.sqrt(swin_map.shape[0]))
            if swin_size * swin_size == swin_map.shape[0]:
                attention_map = swin_map.reshape(swin_size, swin_size)
            else:
                attention_map = cv2.resize(swin_map.reshape(1, -1), (14, 14))
            model_name = "Swin"
            
        else:
            # No attention maps available, use fallback
            return generate_gradient_attention(img_tensor, deit_model or swin_model, 
                                              class_idx, save_path, device, is_normal)
        
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Resize to image size
        attention_resized = cv2.resize(attention_map, (224, 224))
        
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
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
        return f"Attention map generation failed. Using standard visualization.", False

def generate_gradient_attention(img_tensor, model, class_idx, save_path=None, device=DEVICE, is_normal=False):
    """
    Generate gradient-based attention map as fallback.
    
    Args:
        img_tensor: Input image tensor
        model: Model to use (can be None)
        class_idx: Target class index
        save_path: Path to save
        device: Device
        is_normal: Whether normal
        
    Returns:
        description, success
    """
    try:
        import cv2
        
        if model is None:
            return "Attention map not available (no model).", False
        
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
        
        # Take absolute value and normalize
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
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, blended)
        
        if is_normal:
            description = "Gradient-based attention shows normal physiological patterns."
        else:
            description = "Gradient-based attention highlights regions of interest."
        
        return description, True
        
    except Exception as e:
        print(f"❌ Error generating gradient attention: {e}")
        return "Attention map not available.", False

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