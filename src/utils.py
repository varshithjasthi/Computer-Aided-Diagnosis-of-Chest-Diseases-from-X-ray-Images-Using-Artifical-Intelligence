"""
Utility module.
Contains constants, configuration, medical logic, and helper functions.
"""

import os
import torch
from reportlab.lib import colors

# ================= INSTITUTION CONFIG =================
HOSPITAL_NAME = "METRO ADVANCED RADIOLOGY CENTER"
HEAD_RADIOLOGIST = "Dr. Alexander V. Sterling, MD, PhD"
LICENSE_NO = "MC-99203-XRAY"
DEPT = "Division of Chest Imaging & AI Diagnostics"

# ================= PATH CONFIG =================
UPLOAD_FOLDER = "uploads"
HEATMAP_FOLDER = "static/heatmaps"
REPORT_FOLDER = "static/reports"
GRAPH_FOLDER = "static/graphs"

# Create folders automatically if missing
for folder in [UPLOAD_FOLDER, HEATMAP_FOLDER, REPORT_FOLDER, GRAPH_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ================= MODEL CONFIG =================
MODEL_MODE = "hybrid"  # Options: "deit", "swin", "hybrid"
DEIT_MODEL_PATH = os.path.join("models", "deit_small_15_classes.pth")
SWIN_MODEL_PATH = os.path.join("models", "swin_ultra_fast_15_classes.pth")
IMG_SIZE = 224

# Device configuration with startup logging
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🏥 Institution: {HOSPITAL_NAME}")
print(f"🤖 Model Mode: {MODEL_MODE.upper()}")
print(f"🧠 Device: {str(DEVICE).upper()}")

# ================= MEDICAL DATA =================
CLINICAL_SYMPTOMS = [
    "Fever", "Cough", "Shortness of breath", "Chest pain",
    "Fatigue", "Weight loss", "Hemoptysis", "Night sweats"
]

MEDICAL_HISTORY = [
    "Diabetes", "Hypertension", "Asthma", "COPD",
    "Heart Disease", "Kidney Disease", "Liver Disease", "Cancer History"
]

# Enhanced Disease-specific medical explanations
DISEASE_EXPLANATIONS = {
    "NORMAL": {
        "medical_reason": "No radiological abnormalities detected. Anatomical structures appear within normal limits.",
        "ai_reason": "The AI detected no pathological patterns, densities, or structural deviations from normal chest radiograph anatomy.",
        "cause_analysis": {
            "possible_causes": ["Normal anatomical variation", "No active disease process"],
            "risk_factors": ["None identified"]
        },
        "precautions": {
            "do": ["Continue routine health monitoring", "Consult physician if symptoms persist or worsen", "Maintain regular follow-up as advised"],
            "dont": ["Undergo unnecessary imaging", "Assume serious pathology without clinical correlation", "Delay seeking care if new symptoms develop"]
        },
        "lifestyle_recommendations": {
            "diet": ["Balanced diet rich in fruits and vegetables", "Adequate hydration", "Limit processed foods"],
            "exercise": ["Regular moderate exercise 30 minutes/day", "Deep breathing exercises", "Maintain healthy weight"],
            "habits": ["Avoid smoking and secondhand smoke", "Limit alcohol consumption", "Practice good hygiene"]
        },
        "emergency_signs": ["Severe chest pain", "Sudden shortness of breath", "High fever with chills", "Coughing up blood"]
    },
    "Pneumothorax": {
        "medical_reason": "Air in the pleural space causing partial or complete lung collapse, visible as a visceral pleural line with absent lung markings peripheral to it.",
        "ai_reason": "The model detected a sharp pleural line with hyperlucent peripheral area lacking vascular markings.",
        "cause_analysis": {
            "possible_causes": ["Trauma to chest", "Spontaneous rupture of blebs", "Medical procedures", "Underlying lung disease"],
            "risk_factors": ["Tall, thin body habitus", "Smoking", "Family history", "COPD", "Marfan syndrome"]
        },
        "precautions": {
            "do": ["Seek immediate emergency evaluation", "Remain in upright position if possible", "Avoid straining or Valsalva maneuvers"],
            "dont": ["Travel by air", "Perform strenuous activity", "Delay medical attention", "Smoke or vape"]
        },
        "lifestyle_recommendations": {
            "diet": ["Small, frequent meals to reduce strain", "High-protein foods for healing", "Adequate hydration"],
            "exercise": ["Avoid heavy lifting for 4-6 weeks", "Gradual return to activity", "Breathing exercises as advised"],
            "habits": ["Complete smoking cessation", "Avoid pressure changes (diving, flying)", "Use seat belts properly"]
        },
        "emergency_signs": ["Sudden severe chest pain", "Rapid worsening of shortness of breath", "Blue lips or fingers", "Rapid heart rate"]
    },
    "Effusion": {
        "medical_reason": "Abnormal accumulation of fluid in the pleural cavity, typically blunting the costophrenic angle.",
        "ai_reason": "Increased density in the pleural space with meniscus sign and obscured diaphragmatic contour.",
        "cause_analysis": {
            "possible_causes": ["Heart failure", "Pneumonia", "Cancer", "Kidney disease", "Autoimmune disorders"],
            "risk_factors": ["Congestive heart failure", "Liver cirrhosis", "Malignancy", "Recent chest infection"]
        },
        "precautions": {
            "do": ["Complete diagnostic workup for etiology", "Monitor respiratory status closely", "Follow up with pulmonologist"],
            "dont": ["Ignore worsening dyspnea", "Self-medicate without diagnosis", "Assume it will resolve spontaneously"]
        },
        "lifestyle_recommendations": {
            "diet": ["Low-sodium diet", "Fluid restriction if advised", "High-potassium foods", "Limit alcohol"],
            "exercise": ["Light activity as tolerated", "Elevate head during sleep", "Monitor weight daily"],
            "habits": ["Take medications as prescribed", "Monitor for swelling", "Avoid excessive fluid intake"]
        },
        "emergency_signs": ["Severe shortness of breath", "Chest pain with fever", "Coughing up blood", "Confusion or dizziness"]
    },
    "Pneumonia": {
        "medical_reason": "Alveolar consolidation with air bronchograms, indicating inflammatory exudate filling air spaces.",
        "ai_reason": "Patchy or lobar consolidation with air bronchograms and ill-defined margins.",
        "cause_analysis": {
            "possible_causes": ["Bacterial infection", "Viral infection", "Fungal infection", "Aspiration"],
            "risk_factors": ["Age >65 or <2 years", "Smoking", "Chronic lung disease", "Weakened immune system", "Recent hospitalization"]
        },
        "precautions": {
            "do": ["Complete prescribed antibiotic course", "Maintain hydration and rest", "Monitor temperature and oxygen saturation"],
            "dont": ["Discontinue antibiotics early", "Ignore fever recurrence", "Expose vulnerable individuals"]
        },
        "lifestyle_recommendations": {
            "diet": ["Warm fluids (soup, tea)", "High-protein foods", "Fruits rich in Vitamin C", "Avoid dairy if it increases mucus"],
            "exercise": ["Rest until fever subsides", "Gradual return to activity", "Deep breathing exercises"],
            "habits": ["Practice good hand hygiene", "Cover coughs and sneezes", "Get annual flu vaccine", "Consider pneumococcal vaccine"]
        },
        "emergency_signs": ["Difficulty breathing", "Confusion or disorientation", "Persistent high fever", "Blue lips or nails"]
    },
    "Cardiomegaly": {
        "medical_reason": "Enlarged cardiac silhouette, with cardiothoracic ratio exceeding 0.5 in posteroanterior view.",
        "ai_reason": "Increased cardiac silhouette size relative to thoracic diameter.",
        "cause_analysis": {
            "possible_causes": ["Hypertension", "Heart valve disease", "Cardiomyopathy", "Coronary artery disease"],
            "risk_factors": ["High blood pressure", "Family history", "Obesity", "Diabetes", "Sleep apnea"]
        },
        "precautions": {
            "do": ["Cardiology consultation", "Echocardiogram for assessment", "Monitor for heart failure symptoms"],
            "dont": ["Ignore associated symptoms", "Discontinue cardiac medications", "Engage in intense exertion without clearance"]
        },
        "lifestyle_recommendations": {
            "diet": ["Low-sodium diet", "Heart-healthy fats", "Whole grains", "Limit processed foods"],
            "exercise": ["Regular moderate exercise", "Avoid heavy weight lifting", "Monitor exercise tolerance"],
            "habits": ["Regular blood pressure monitoring", "Weight management", "Stress reduction techniques"]
        },
        "emergency_signs": ["Severe chest pain", "Sudden shortness of breath", "Fainting or near-fainting", "Irregular heart rhythm"]
    },
    "Atelectasis": {
        "medical_reason": "Partial or complete lung collapse due to airway obstruction or compression.",
        "ai_reason": "Increased density with volume loss and displacement of fissures.",
        "cause_analysis": {
            "possible_causes": ["Mucus plugging", "Foreign body", "Tumor compression", "Post-surgical"],
            "risk_factors": ["Recent surgery", "Shallow breathing", "Smoking", "Neuromuscular disorders"]
        },
        "precautions": {
            "do": ["Pulmonary toilet exercises", "Deep breathing exercises", "Address underlying cause"],
            "dont": ["Remain sedentary", "Ignore worsening dyspnea", "Smoke or expose to irritants"]
        },
        "lifestyle_recommendations": {
            "diet": ["Adequate hydration to thin secretions", "Balanced nutrition", "Avoid large meals before bed"],
            "exercise": ["Incentive spirometry", "Walking as tolerated", "Postural drainage techniques"],
            "habits": ["Smoking cessation", "Practice deep breathing", "Change positions frequently"]
        },
        "emergency_signs": ["Sudden severe shortness of breath", "Chest pain with breathing", "High fever", "Blue skin discoloration"]
    },
    "Infiltration": {
        "medical_reason": "Increased lung density with irregular margins, suggesting inflammatory process.",
        "ai_reason": "The model detected patchy areas of increased density without clear borders.",
        "cause_analysis": {
            "possible_causes": ["Early pneumonia", "Inflammatory conditions", "Drug reaction", "Radiation injury"],
            "risk_factors": ["Recent infection", "Autoimmune disease", "Immunosuppression", "Environmental exposures"]
        },
        "precautions": {
            "do": ["Consult pulmonologist", "Consider infectious workup", "Monitor for symptom progression"],
            "dont": ["Ignore persistent symptoms", "Self-treat without diagnosis", "Delay follow-up imaging"]
        },
        "lifestyle_recommendations": {
            "diet": ["Anti-inflammatory foods", "Adequate protein", "Foods rich in antioxidants"],
            "exercise": ["Light activity as tolerated", "Avoid strenuous exercise during acute phase"],
            "habits": ["Avoid respiratory irritants", "Use air purifier if needed", "Practice good hygiene"]
        },
        "emergency_signs": ["Worsening shortness of breath", "High fever unresponsive to medication", "Chest pain", "Confusion"]
    },
    "Mass": {
        "medical_reason": "Discrete, space-occupying lesion with defined margins, requiring further characterization.",
        "ai_reason": "Focal rounded density with distinct borders detected by the model.",
        "cause_analysis": {
            "possible_causes": ["Primary lung cancer", "Metastatic cancer", "Benign tumor", "Infection"],
            "risk_factors": ["Smoking", "Family history of cancer", "Radon exposure", "Occupational exposures"]
        },
        "precautions": {
            "do": ["Urgent oncology/pulmonology referral", "CT scan for further characterization", "Biopsy if indicated"],
            "dont": ["Delay evaluation", "Assume benign without workup", "Ignore size changes on follow-up"]
        },
        "lifestyle_recommendations": {
            "diet": ["Cancer-preventive diet", "High-fiber foods", "Cruciferous vegetables", "Limit red meat"],
            "exercise": ["Maintain physical activity as tolerated", "Consult doctor for exercise limits"],
            "habits": ["Complete smoking cessation", "Avoid environmental carcinogens", "Regular medical follow-up"]
        },
        "emergency_signs": ["Severe chest pain", "Coughing up large amounts of blood", "Sudden shortness of breath", "Unintended weight loss >10%"]
    },
    "Nodule": {
        "medical_reason": "Small, rounded opacity less than 3 cm in diameter, requiring follow-up for stability.",
        "ai_reason": "Small focal density detected, assessed for malignancy risk factors.",
        "cause_analysis": {
            "possible_causes": ["Old granuloma", "Early cancer", "Benign growth", "Inflammatory"],
            "risk_factors": ["Smoking history", "Age >50", "Family history of lung cancer", "Occupational exposures"]
        },
        "precautions": {
            "do": ["Follow Fleischner Society guidelines", "CT scan for better characterization", "Document for follow-up"],
            "dont": ["Panic - most nodules are benign", "Ignore recommended follow-up", "Assume malignancy without workup"]
        },
        "lifestyle_recommendations": {
            "diet": ["Antioxidant-rich foods", "Foods with Vitamin D", "Omega-3 fatty acids"],
            "exercise": ["Regular cardiovascular exercise", "Strength training", "Maintain healthy weight"],
            "habits": ["Smoking cessation if applicable", "Avoid secondhand smoke", "Regular health screenings"]
        },
        "emergency_signs": ["Growth on follow-up scan", "Development of new symptoms", "Coughing up blood", "Unexplained weight loss"]
    },
    "COVID19": {
        "medical_reason": "Bilateral peripheral ground-glass opacities and consolidations, typical of viral pneumonia pattern.",
        "ai_reason": "The model detected bilateral peripheral lung opacities with ground-glass appearance.",
        "cause_analysis": {
            "possible_causes": ["SARS-CoV-2 viral infection", "Post-COVID inflammatory response"],
            "risk_factors": ["Unvaccinated status", "Advanced age", "Immunocompromised state", "Comorbid conditions"]
        },
        "precautions": {
            "do": ["Consult infectious disease specialist", "Isolate to prevent transmission", "Monitor oxygen saturation", "Follow local health guidelines"],
            "dont": ["Self-medicate without medical advice", "Delay seeking care if symptoms worsen", "Ignore public health recommendations"]
        },
        "lifestyle_recommendations": {
            "diet": ["High-protein diet for recovery", "Vitamin C and D supplementation", "Adequate hydration", "Small frequent meals"],
            "exercise": ["Rest during acute phase", "Gradual return to activity", "Breathing exercises", "Avoid strenuous exertion"],
            "habits": ["Practice strict respiratory hygiene", "Get COVID-19 vaccination", "Wear masks in crowded settings", "Regular hand washing"]
        },
        "emergency_signs": ["Severe shortness of breath", "Oxygen saturation <92%", "Confusion or lethargy", "Persistent high fever", "Chest pain"]
    },
    "Consolidation": {
        "medical_reason": "Homogeneous increase in lung density obscuring underlying vessels, often with air bronchograms.",
        "ai_reason": "The model detected areas of homogeneous opacity with loss of normal lung markings.",
        "cause_analysis": {
            "possible_causes": ["Bacterial pneumonia", "Pulmonary edema", "Hemorrhage", "Neoplasm"],
            "risk_factors": ["Recent respiratory infection", "Aspiration risk", "Immunosuppression", "Heart failure"]
        },
        "precautions": {
            "do": ["Complete diagnostic workup", "Start appropriate antibiotics if infectious", "Monitor respiratory status", "Consider CT scan if unresolved"],
            "dont": ["Discontinue treatment early", "Ignore fever or worsening symptoms", "Delay follow-up imaging"]
        },
        "lifestyle_recommendations": {
            "diet": ["Adequate protein for healing", "Vitamin-rich foods", "Warm fluids", "Avoid dairy if increases mucus"],
            "exercise": ["Rest during acute phase", "Incentive spirometry", "Gradual ambulation as tolerated"],
            "habits": ["Smoking cessation", "Practice deep breathing", "Elevate head during sleep"]
        },
        "emergency_signs": ["Severe dyspnea", "High fever with chills", "Hypotension", "Altered mental status"]
    },
    "Edema": {
        "medical_reason": "Increased interstitial markings, peribronchial cuffing, and Kerley lines indicating fluid in lung interstitium.",
        "ai_reason": "The model detected increased interstitial markings and vascular redistribution pattern.",
        "cause_analysis": {
            "possible_causes": ["Heart failure", "Renal failure", "Fluid overload", "ARDS", "High altitude"],
            "risk_factors": ["Congestive heart failure", "Kidney disease", "Excessive IV fluids", "Pulmonary hypertension"]
        },
        "precautions": {
            "do": ["Urgent cardiology/nephrology consultation", "Diuretic therapy as prescribed", "Monitor fluid intake/output", "Daily weight monitoring"],
            "dont": ["Ignore worsening shortness of breath", "Discontinue medications without advice", "Consume high-sodium foods"]
        },
        "lifestyle_recommendations": {
            "diet": ["Strict sodium restriction", "Fluid restriction as advised", "Potassium-rich foods if on diuretics", "Small frequent meals"],
            "exercise": ["Activity as tolerated", "Elevate legs when sitting", "Avoid strenuous exertion", "Monitor exercise tolerance"],
            "habits": ["Take medications exactly as prescribed", "Weigh daily at same time", "Monitor for swelling", "Avoid alcohol"]
        },
        "emergency_signs": ["Severe breathing difficulty", "Pink frothy sputum", "Chest pain", "Confusion or dizziness", "Rapid weight gain"]
    },
    "Tuberculosis": {
        "medical_reason": "Upper lobe predominant infiltrates, cavities, and nodules suggesting mycobacterial infection.",
        "ai_reason": "The model detected apical infiltrates and cavitary lesions typical of mycobacterial infection.",
        "cause_analysis": {
            "possible_causes": ["Mycobacterium tuberculosis infection", "Reactivation of latent TB"],
            "risk_factors": ["Immunocompromised state", "HIV infection", "Diabetes", "Close contact with active TB", "Malnutrition"]
        },
        "precautions": {
            "do": ["Immediate infectious disease consultation", "Respiratory isolation until non-infectious", "Complete full course of anti-TB therapy", "Screen close contacts"],
            "dont": ["Discontinue medications prematurely", "Ignore public health notifications", "Delay treatment initiation"]
        },
        "lifestyle_recommendations": {
            "diet": ["High-calorie, high-protein diet", "Vitamin supplementation", "Adequate hydration", "Nutrient-dense foods"],
            "exercise": ["Rest during active phase", "Gradual return to activity", "Deep breathing exercises", "Avoid overexertion"],
            "habits": ["Practice respiratory hygiene", "Take medications exactly as prescribed", "Attend all follow-up appointments", "Avoid smoking and alcohol"]
        },
        "emergency_signs": ["Massive hemoptysis", "Severe chest pain", "High fever unresponsive to medication", "Difficulty breathing", "Signs of TB meningitis"]
    }
}

# Default explanation for unspecified diseases
DEFAULT_EXPLANATION = {
    "medical_reason": "Abnormal radiographic finding requiring clinical correlation.",
    "ai_reason": "The AI detected patterns inconsistent with normal anatomy.",
    "cause_analysis": {
        "possible_causes": ["Requires further investigation", "Multiple possible etiologies"],
        "risk_factors": ["Clinical correlation needed"]
    },
    "precautions": {
        "do": ["Consult with specialist", "Complete recommended workup", "Monitor for symptom progression"],
        "dont": ["Ignore the finding", "Self-diagnose or treat", "Delay follow-up"]
    },
    "lifestyle_recommendations": {
        "diet": ["Balanced nutrition", "Adequate hydration", "Limit processed foods"],
        "exercise": ["Moderate activity as tolerated", "Consult doctor for specific recommendations"],
        "habits": ["Avoid smoking", "Limit alcohol", "Regular medical check-ups"]
    },
    "emergency_signs": ["Worsening symptoms", "Difficulty breathing", "Severe pain", "High fever"]
}

# ================= UTILITY FUNCTIONS =================

def normalize_class_name(class_name):
    """
    Normalize class names for consistent processing.
    Maps various forms of "normal" and "no finding" to "NORMAL".
    """
    if not class_name:
        return "NORMAL"
    
    upper_name = class_name.upper().strip()
    
    # Check for any variation of normal or no finding
    normal_patterns = ["NORMAL", "NO FINDING", "NOFINDING", "NO FINDINGS", "NOFINDINGS"]
    if any(pattern in upper_name for pattern in normal_patterns):
        return "NORMAL"
    
    return class_name

def get_risk_assessment(class_name, probability):
    """Returns risk assessment based on class and probability."""
    normalized_class = normalize_class_name(class_name)
    
    if normalized_class == "NORMAL":
        return {
            "risk_level": "NO RISK",
            "display_text": "NORMAL RADIOGRAPH",
            "color": colors.green,
            "hex_color": "#10b981",
            "is_normal": True
        }
    
    # Only disease classes get risk levels
    # HIGH RISK: probability >= 60
    # MODERATE RISK: probability >= 30
    # LOW RISK: probability < 30
    if probability >= 60:
        return {
            "risk_level": "HIGH RISK",
            "display_text": "HIGH CLINICAL CONCERN",
            "color": colors.red,
            "hex_color": "#ef4444",
            "is_normal": False
        }
    elif probability >= 30:
        return {
            "risk_level": "MODERATE RISK",
            "display_text": "MODERATE CLINICAL CONCERN",
            "color": colors.orange,
            "hex_color": "#f59e0b",
            "is_normal": False
        }
    else:
        return {
            "risk_level": "LOW RISK",
            "display_text": "LOW CLINICAL CONCERN",
            "color": colors.dodgerblue,
            "hex_color": "#3b82f6",
            "is_normal": False
        }

def get_disease_explanation(disease_name):
    """Returns disease-specific explanation or default if disease not found."""
    normalized_name = normalize_class_name(disease_name)
    return DISEASE_EXPLANATIONS.get(normalized_name, DEFAULT_EXPLANATION)

def get_specialist_for_disease(disease):
    """Returns appropriate specialist for disease follow-up."""
    specialties = {
        "PNEUMOTHORAX": "pulmonologist or thoracic surgeon",
        "PNEUMONIA": "pulmonologist or infectious disease specialist",
        "EFFUSION": "pulmonologist",
        "CARDIOMEGALY": "cardiologist",
        "ATELECTASIS": "pulmonologist",
        "MASS": "oncologist or pulmonologist",
        "NODULE": "pulmonologist",
        "INFILTRATION": "pulmonologist",
        "COVID19": "pulmonologist or infectious disease specialist",
        "CONSOLIDATION": "pulmonologist",
        "EDEMA": "cardiologist or pulmonologist",
        "TUBERCULOSIS": "infectious disease specialist or pulmonologist"
    }
    return specialties.get(disease.upper(), "appropriate specialist")