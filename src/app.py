"""
Main Flask application entry point.
Handles routing, request processing, and error handling.
Complete AI Radiology Assistant with Hybrid Transformer Architecture.
"""

import os
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
from dotenv import load_dotenv
import imghdr

# Import from models
from models import load_models, hybrid_probability_fusion, generate_hybrid_attention_map

# Import from services
from services import (
    process_image, generate_medical_graphs, generate_detailed_pdf, get_ai_guidance,
    validate_image_format
)

# Import from utils
from utils import (
    UPLOAD_FOLDER, HEATMAP_FOLDER, REPORT_FOLDER, GRAPH_FOLDER,
    HOSPITAL_NAME, HEAD_RADIOLOGIST, DEPT, LICENSE_NO,
    CLINICAL_SYMPTOMS, MEDICAL_HISTORY,
    get_risk_assessment, get_disease_explanation, normalize_class_name,
    DEVICE, MODEL_MODE
)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_UPLOAD_SIZE', 16 * 1024 * 1024))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Create necessary directories
for folder in [UPLOAD_FOLDER, HEATMAP_FOLDER, REPORT_FOLDER, GRAPH_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Allowed file extensions for validation
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load AI models (hybrid architecture)
print(f"\n{'='*60}")
print(f"🏥 METRO ADVANCED RADIOLOGY CENTER - AI DIAGNOSTIC SYSTEM")
print(f"{'='*60}")

# Load models - FIXED: Correct unpacking (2 models, not 3)
hybrid_model, deit_model, swin_model, class_names = load_models()
print(f"\n{'='*60}")
print(f"✅ System Ready - Active Mode: {MODEL_MODE.upper()}")
print(f"{'='*60}\n")

# ================= FLASK ROUTES =================

@app.route("/")
def index():
    """Render main interface with hospital branding."""
    return render_template("index.html", 
                         hospital=HOSPITAL_NAME, 
                         doctor=HEAD_RADIOLOGIST,
                         symptoms=CLINICAL_SYMPTOMS,
                         medical_history=MEDICAL_HISTORY,
                         model_mode=MODEL_MODE)

@app.route("/uploads/<filename>")
def serve_upload(filename):
    """Serve uploaded images from the upload folder."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/static/heatmaps/<filename>")
def serve_heatmap(filename):
    """Serve generated heatmap images."""
    return send_from_directory(HEATMAP_FOLDER, filename)

@app.route("/static/graphs/<filename>")
def serve_graph(filename):
    """Serve generated graph images."""
    return send_from_directory(GRAPH_FOLDER, filename)

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main analysis endpoint.
    Processes radiographic images through hybrid AI models and generates comprehensive reports.
    """
    try:
        print("📤 Image received - starting analysis pipeline")
        
        # 🔥 CRITICAL: Check content length BEFORE accessing request.form
        max_size = int(os.getenv('MAX_UPLOAD_SIZE', 16 * 1024 * 1024))
        if request.content_length and request.content_length > max_size:
            print(f"❌ Request too large: {request.content_length} bytes")
            return jsonify({
                "status": "error",
                "message": f"Request too large. Maximum image size is {max_size // (1024*1024)}MB. Please use a smaller image or compress it."
            }), 413
        
        # Extract patient information from form
        patient = {
            "id": request.form.get("patient_id") or f"PX-{datetime.now().strftime('%M%S')}",
            "name": request.form.get("patient_name") or "Patient-Unknown",
            "age": request.form.get("age") or "N/A",
            "sex": request.form.get("sex") or "U",
            "date": request.form.get("exam_date") or datetime.now().strftime("%Y-%m-%d"),
            "referring_dr": request.form.get("referring_dr") or "Not Specified"
        }
        
        # Extract clinical data
        symptoms = request.form.getlist("symptoms[]")
        medical_history = request.form.getlist("medical_history[]")
        patient_vitals = {
            "bp": request.form.get("blood_pressure") or "Not recorded",
            "sugar": request.form.get("sugar_level") or "Not recorded",
            "smoking": request.form.get("smoking") or "Not recorded",
            "alcohol": request.form.get("alcohol") or "Not recorded"
        }
        
        # Get image data (base64 from camera or file upload)
        cam_data = request.form.get("captured_image")
        upload_file = request.files.get("image")
        
        # Validate image input
        if not cam_data and (not upload_file or upload_file.filename == ''):
            print("❌ No image provided")
            return jsonify({
                "status": "error", 
                "message": "No radiographic image provided. Please capture or upload an image."
            }), 400

        # Validate file type for uploads
        if upload_file and upload_file.filename:
            if not allowed_file(upload_file.filename):
                print(f"❌ Invalid file type: {upload_file.filename}")
                return jsonify({
                    "status": "error",
                    "message": "Invalid file type. Please upload JPG, JPEG, or PNG images only."
                }), 400

        # Generate unique filename for this analysis
        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"{patient['id']}_{timestamp}"
        save_path = os.path.join(UPLOAD_FOLDER, filename + ".jpg")

        # Process and save the image
        print("💾 Saving image...")
        image_saved = process_image(cam_data, upload_file, save_path)
        if not image_saved:
            print("❌ Failed to process image")
            return jsonify({
                "status": "error", 
                "message": "Failed to process image. The image may be too large or corrupted."
            }), 400
        
        # Validate image format after saving
        if not validate_image_format(save_path):
            print("❌ Invalid image format")
            os.remove(save_path)
            return jsonify({
                "status": "error",
                "message": "Invalid image format. Please upload a valid JPG, JPEG, or PNG image."
            }), 400
        
        # Run AI inference using hybrid model architecture
        print(f"🧠 Running hybrid inference with {MODEL_MODE.upper()} mode...")
        try:
            predictions, img_tensor = hybrid_probability_fusion(
    image_path=save_path,
    hybrid_model=hybrid_model,
    deit_model=deit_model,
    swin_model=swin_model,
    class_names=class_names,
    mode=MODEL_MODE,
    device=DEVICE
)
            print("✅ Inference complete")
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return jsonify({
                "status": "error",
                "message": "AI model inference failed. Please try again."
            }), 500
        
        # Determine primary finding
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        primary_prediction = predictions[0]
        primary_class = normalize_class_name(primary_prediction["disease"])
        is_normal = primary_class == "NORMAL"
        
        # Calculate probability distributions
        normal_pred = next((p for p in predictions if normalize_class_name(p["disease"]) == "NORMAL"), None)
        normal_percentage = normal_pred["probability"] if normal_pred else 0
        disease_predictions = [p for p in predictions if normalize_class_name(p["disease"]) != "NORMAL"]
        disease_percentage = sum(p["probability"] for p in disease_predictions)
        
        # Create disease breakdown with contributions
        disease_breakdown = []
        for pred in disease_predictions:
            if disease_percentage > 0:
                contribution = (pred["probability"] / disease_percentage) * 100
            else:
                contribution = 0
            disease_breakdown.append({
                "disease": pred["disease"],
                "probability": pred["probability"],
                "contribution": round(contribution, 2)
            })
        disease_breakdown.sort(key=lambda x: x["contribution"], reverse=True)
        
        # Get risk assessment for primary finding
        risk_info = get_risk_assessment(primary_class, primary_prediction["probability"])
        
        # Generate AI attention map (heatmap)
        print("🔥 Generating attention map...")
        primary_class_idx = class_names.index(primary_prediction["disease"]) if primary_prediction["disease"] in class_names else 0
        h_name = f"heat_{filename}.jpg"
        h_path = os.path.join(HEATMAP_FOLDER, h_name)
        
        try:
            heatmap_description, heatmap_success = generate_hybrid_attention_map(
                img_tensor=img_tensor,
                deit_model=deit_model,
                swin_model=swin_model,
                class_idx=primary_class_idx,
                class_names=class_names,
                save_path=h_path,
                device=DEVICE,
                mode=MODEL_MODE,
                is_normal=is_normal
            )
            print(f"✅ Heatmap generated: {heatmap_success}")
        except Exception as e:
            print(f"⚠️ Heatmap generation failed: {e}")
            heatmap_description = "Attention map generation unavailable."
            heatmap_success = False
        
        # Generate medical visualization graphs
        print("📊 Generating medical graphs...")
        try:
            graphs = generate_medical_graphs(predictions, filename)
            print("✅ Graphs generated")
        except Exception as e:
            print(f"⚠️ Graph generation failed: {e}")
            graphs = {"donut_chart": None, "bar_chart": None, "risk_chart": None}
        
        # Get AI-generated clinical guidance from OpenRouter
        print("🤖 Requesting AI guidance...")
        try:
            ai_guidance = get_ai_guidance(
                disease=primary_class,
                probability=primary_prediction["probability"],
                symptoms=symptoms,
                patient_vitals=patient_vitals
            )
            print("✅ AI guidance received")
        except Exception as e:
            print(f"⚠️ AI guidance failed: {e}")
            explanation = get_disease_explanation(primary_class)
            ai_guidance = {
                "explanation": explanation["medical_reason"],
                "diet": "; ".join(explanation["lifestyle_recommendations"]["diet"]),
                "lifestyle": "; ".join(explanation["lifestyle_recommendations"]["habits"]),
                "warning_signs": "; ".join(explanation["emergency_signs"]),
                "disclaimer": "AI guidance unavailable. Using standard medical guidance."
            }
        
        # Generate comprehensive PDF report
        print("📄 Generating PDF report...")
        pdf_name = f"Radiology_Report_{patient['id']}_{timestamp}.pdf"
        pdf_path = os.path.join(REPORT_FOLDER, pdf_name)
        
        try:
            generate_detailed_pdf(
                pdf_path=pdf_path,
                patient=patient,
                image_path=save_path,
                heatmap_path=h_path if heatmap_success else None,
                predictions=predictions,
                symptoms=symptoms,
                patient_vitals=patient_vitals,
                medical_history=medical_history,
                is_normal=is_normal,
                heatmap_description=heatmap_description,
                ai_guidance=ai_guidance
            )
            print("✅ PDF report generated")
        except Exception as e:
            print(f"❌ PDF generation failed: {e}")
            # Continue without PDF - we'll still return results

        # Prepare comprehensive response data
        response_data = {
            "status": "success",
            "primary_finding": primary_prediction["disease"],
            "primary_probability": primary_prediction["probability"],
            "is_normal": is_normal,
            "normal_percentage": round(normal_percentage, 2),
            "disease_percentage": round(disease_percentage, 2),
            "disease_breakdown": disease_breakdown[:5],  # Top 5 contributions
            "graphs": graphs,
            "risk_assessment": {
                "level": risk_info["risk_level"],
                "display_text": risk_info["display_text"],
                "color": risk_info["hex_color"]
            },
            "predictions": [
                {
                    "disease": p["disease"],
                    "probability": p["probability"],
                    "risk": get_risk_assessment(p["disease"], p["probability"])["risk_level"],
                    "cause_analysis": get_disease_explanation(p["disease"])["cause_analysis"],
                    "lifestyle_recommendations": get_disease_explanation(p["disease"])["lifestyle_recommendations"]
                } 
                for p in predictions[:5]  # Top 5 predictions with full details
            ],
            "symptoms": symptoms,
            "patient_vitals": patient_vitals,
            "medical_history": medical_history,
            "image_url": f"/uploads/{filename}.jpg",
            "pdf_url": f"/download?path={pdf_path}" if os.path.exists(pdf_path) else None,
            "report_id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "heatmap_description": heatmap_description,
            "ai_guidance": ai_guidance,
            "heatmap_available": heatmap_success,
            "heatmap_url": f"/static/heatmaps/{h_name}" if heatmap_success else None,
            "show_heatmap": not is_normal and heatmap_success,
            "model_mode": MODEL_MODE,
            "model_info": {
                "deit": "Online",
                "swin": "Online",
                "fusion": "Weighted Average (50/50)"
            }
        }

        # Add clinical note based on findings
        if is_normal:
            response_data["clinical_note"] = "No abnormality detected. Chest radiograph is within normal limits."
            if symptoms:
                response_data["clinical_note"] += f" Reported symptoms ({', '.join(symptoms)}) require clinical correlation as they may be non-radiographic."
        else:
            explanation = get_disease_explanation(primary_class)
            response_data["clinical_note"] = f"Findings suggestive of {primary_class}. {explanation['medical_reason']}"
            if symptoms:
                response_data["clinical_note"] += f" Symptoms are consistent with this finding and increase clinical suspicion."

        print(f"✅ Analysis complete for patient {patient['id']}")
        return jsonify(response_data)

    except Exception as e:
        print(f"🚨 SYSTEM ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": f"System error during analysis: {str(e)[:200]}",
            "clinical_note": "Analysis incomplete. Please retry or consult technical support."
        }), 500

@app.route("/download")
def download():
    """Download generated PDF report."""
    try:
        path = request.args.get("path")
        if not path or not os.path.exists(path):
            return jsonify({"status": "error", "message": "Report not found"}), 404
        return send_file(path, as_attachment=True, download_name=os.path.basename(path))
    except Exception as e:
        print(f"❌ Download error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/symptoms")
def get_symptoms():
    """API endpoint to get available symptom options for UI."""
    return jsonify({
        "symptoms": CLINICAL_SYMPTOMS, 
        "medical_history": MEDICAL_HISTORY
    })

@app.route("/api/model-info")
def model_info():
    """API endpoint to get current model configuration."""
    return jsonify({
        "mode": MODEL_MODE,
        "device": str(DEVICE),
        "classes": class_names,
        "models": {
            "deit": "Online",
            "swin": "Online",
            "hybrid": "Active" if MODEL_MODE == "hybrid" else "Standby"
        }
    })

@app.route("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_mode": MODEL_MODE,
        "device": str(DEVICE),
        "openrouter": "Configured" if os.getenv('OPENROUTER_API_KEY') else "Not configured"
    })

@app.route("/api/health")
def api_health():
    """API health check endpoint (alias for /health)."""
    return health_check()

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return jsonify({
        "status": "error",
        "message": "Resource not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 internal server errors."""
    return jsonify({
        "status": "error",
        "message": "Internal server error. Please try again later."
    }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file upload size limit errors."""
    max_size = int(os.getenv('MAX_UPLOAD_SIZE', 16 * 1024 * 1024))
    return jsonify({
        "status": "error",
        "message": f"File too large. Maximum size is {max_size // (1024*1024)}MB. Please compress the image or use a smaller file."
    }), 413

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🚀 NEURORAD AI - Hybrid Transformer System")
    print(f"{'='*60}")
    print(f"📊 Model Mode: {MODEL_MODE.upper()}")
    print(f"🤖 Neural Architectures: DeiT-Small + Swin-Tiny")
    print(f"📈 Probability Fusion: Weighted Averaging (50% DeiT + 50% Swin)")
    print(f"🏥 Institution: {HOSPITAL_NAME}")
    print(f"👨‍⚕️ Lead Radiologist: {HEAD_RADIOLOGIST}")
    print(f"💻 Device: {DEVICE}")
    print(f"⚡ Max upload size: {int(os.getenv('MAX_UPLOAD_SIZE', 16 * 1024 * 1024)) // (1024*1024)}MB")
    print(f"🔑 OpenRouter AI: {'✅ Configured' if os.getenv('OPENROUTER_API_KEY') else '⚠️ Not configured - using fallback'}")
    print(f"{'='*60}")
    print(f"\n⚠️  DISCLAIMER: This is an AI-assisted hybrid system, not a diagnostic tool.")
    print(f"    All findings require clinical correlation by qualified professionals.\n")
    
    # Start Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)