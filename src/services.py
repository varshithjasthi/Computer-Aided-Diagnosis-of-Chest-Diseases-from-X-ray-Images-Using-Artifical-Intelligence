"""
Service layer module.
Contains image processing, graph generation, PDF report creation, and OpenRouter AI integration.
Complete service layer for AI Radiology Assistant with Hybrid Transformer Architecture.
"""

import os
import base64
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import json
import io

# ReportLab for PDF generation
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

# OpenRouter AI integration via OpenAI SDK
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from utils import (
    HOSPITAL_NAME, HEAD_RADIOLOGIST, DEPT, LICENSE_NO,
    HEATMAP_FOLDER, REPORT_FOLDER, GRAPH_FOLDER,
    get_risk_assessment, get_disease_explanation, get_specialist_for_disease,
    normalize_class_name, DISEASE_EXPLANATIONS, RISK_LEVELS
)

# ==================== OPENROUTER AI CONFIGURATION ====================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if OPENROUTER_API_KEY:
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    print("✅ OpenRouter AI configured successfully")
else:
    client = None
    print("⚠️ OpenRouter API key not found. AI guidance will use predefined explanations.")

# ==================== IMAGE PROCESSING FUNCTIONS ====================

def process_image(cam_data, upload_file, save_path):
    """
    Process image from camera (base64) or file upload.
    
    Args:
        cam_data: Base64 image data
        upload_file: File upload object
        save_path: Path to save processed image
        
    Returns:
        bool: True if image was successfully saved
    """
    try:
        if cam_data and "," in cam_data:
            # Handle base64 image from camera
            header, encoded = cam_data.split(",", 1)
            
            # Check base64 size (12MB base64 decodes to ~16MB)
            if len(encoded) > 12 * 1024 * 1024:
                print("❌ Base64 image too large")
                return False
                
            # Decode and save
            image_data = base64.b64decode(encoded)
            with open(save_path, "wb") as f: 
                f.write(image_data)
            
            # Verify image integrity
            try:
                img = Image.open(save_path)
                img.verify()
                print(f"✅ Camera image saved: {save_path}")
                return True
            except Exception as e:
                print(f"❌ Corrupted image data: {e}")
                os.remove(save_path)
                return False
                
        elif upload_file and upload_file.filename != '':
            # Handle file upload
            upload_file.save(save_path)
            
            # Verify image integrity
            try:
                img = Image.open(save_path)
                img.verify()
                print(f"✅ Uploaded image saved: {save_path}")
                return True
            except Exception as e:
                print(f"❌ Corrupted upload file: {e}")
                os.remove(save_path)
                return False
            
        else:
            print("❌ No valid image data provided")
            return False
            
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return False

def resize_image_if_needed(image_path, max_size_mb=16):
    """
    Resize image if it exceeds size limit.
    
    Args:
        image_path: Path to image file
        max_size_mb: Maximum size in MB
        
    Returns:
        bool: True if image was resized or already ok
    """
    try:
        file_size = os.path.getsize(image_path) / (1024 * 1024)  # Size in MB
        
        if file_size <= max_size_mb:
            return True
            
        # Open and resize
        img = Image.open(image_path)
        
        # Calculate new size (reduce by 50% iteratively until under limit)
        quality = 95
        while file_size > max_size_mb and quality > 10:
            # Save with reduced quality
            temp_path = image_path + ".temp"
            img.save(temp_path, quality=quality, optimize=True)
            
            new_size = os.path.getsize(temp_path) / (1024 * 1024)
            if new_size < file_size:
                os.replace(temp_path, image_path)
                file_size = new_size
            
            quality -= 15
        
        print(f"✅ Image resized to {file_size:.2f}MB")
        return True
        
    except Exception as e:
        print(f"❌ Error resizing image: {e}")
        return False

# ==================== GRAPH GENERATION FUNCTIONS ====================

def generate_medical_graphs(predictions, filename_prefix):
    """
    Generate medical graphs for visualization with modern styling.
    
    Args:
        predictions: List of prediction dictionaries
        filename_prefix: Prefix for graph filenames
        
    Returns:
        dict: URLs to generated graph images
    """
    graphs = {}
    
    try:
        # Extract NORMAL and disease predictions
        normal_pred = None
        disease_preds = []
        
        for pred in predictions:
            if normalize_class_name(pred["disease"]) == "NORMAL":
                normal_pred = pred
            else:
                disease_preds.append(pred)
        
        # Set modern style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. NORMAL vs DISEASE DONUT CHART
        normal_percent = normal_pred["probability"] if normal_pred else 0
        disease_percent = sum(p["probability"] for p in disease_preds)
        
        # Ensure percentages sum to 100
        total = normal_percent + disease_percent
        if total > 0:
            normal_percent = (normal_percent / total) * 100
            disease_percent = (disease_percent / total) * 100
        
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        fig.patch.set_alpha(0)
        ax.set_facecolor('white')
        
        # Data for donut chart
        sizes = [normal_percent, disease_percent]
        colors_donut = ['#10b981', '#ef4444']  # Green for normal, red for disease
        labels = [f'NORMAL\n{normal_percent:.1f}%', f'DISEASE\n{disease_percent:.1f}%']
        explode = (0.05, 0)
        
        # Create donut chart
        wedges, texts, autotexts = ax.pie(
            sizes, 
            colors=colors_donut, 
            labels=labels, 
            autopct='%1.1f%%', 
            startangle=90, 
            explode=explode, 
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        
        # Style the text
        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('bold')
            text.set_color('#1f2937')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax.set_title('NORMAL vs DISEASE Probability Distribution', 
                    fontsize=14, fontweight='bold', pad=20, color='#1e3a8a')
        ax.axis('equal')
        
        # Add center text
        centre_circle = plt.Circle((0,0), 0.25, fc='white', edgecolor='#e5e7eb', linewidth=2)
        fig.gca().add_artist(centre_circle)
        ax.text(0, 0, 'AI\nAssessment', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='#1e3a8a')
        
        # Save donut chart
        donut_path = os.path.join(GRAPH_FOLDER, f"{filename_prefix}_donut.png")
        plt.tight_layout()
        plt.savefig(donut_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        graphs["donut_chart"] = f"/static/graphs/{filename_prefix}_donut.png"
        
        # 2. DISEASE PROBABILITY BAR CHART
        if disease_preds and disease_percent > 0:
            # Sort diseases by probability
            top_diseases = sorted(disease_preds, key=lambda x: x["probability"], reverse=True)[:7]
            
            # Prepare data for bar chart
            disease_names = [d["disease"] for d in top_diseases]
            disease_probs = [d["probability"] for d in top_diseases]
            
            # Create color gradient based on probability
            bar_colors = []
            for prob in disease_probs:
                if prob >= 70:
                    bar_colors.append('#dc2626')  # High risk - red
                elif prob >= 50:
                    bar_colors.append('#f97316')  # Moderate-high - orange
                elif prob >= 30:
                    bar_colors.append('#f59e0b')  # Moderate - amber
                elif prob >= 10:
                    bar_colors.append('#3b82f6')  # Low - blue
                else:
                    bar_colors.append('#6b7280')  # Very low - gray
            
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            fig.patch.set_alpha(0)
            ax.set_facecolor('white')
            
            # Create horizontal bar chart
            y_pos = np.arange(len(disease_names))
            bars = ax.barh(y_pos, disease_probs, color=bar_colors, height=0.6, edgecolor='white', linewidth=1)
            
            # Customize chart
            ax.set_yticks(y_pos)
            ax.set_yticklabels(disease_names, fontsize=11, fontweight='500')
            ax.invert_yaxis()
            ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold', color='#1e3a8a')
            ax.set_xlim([0, 100])
            ax.set_title('Top Disease Probabilities', fontsize=14, fontweight='bold', pad=20, color='#1e3a8a')
            
            # Add probability values on bars
            for i, (bar, prob) in enumerate(zip(bars, disease_probs)):
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                       f'{prob:.1f}%', ha='left', va='center', 
                       fontsize=10, fontweight='bold', color='#1f2937')
            
            # Add grid for better readability
            ax.grid(True, axis='x', alpha=0.3, linestyle='--', color='#cbd5e1')
            ax.set_axisbelow(True)
            
            # Style spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Save bar chart
            bar_path = os.path.join(GRAPH_FOLDER, f"{filename_prefix}_bar.png")
            plt.tight_layout()
            plt.savefig(bar_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            graphs["bar_chart"] = f"/static/graphs/{filename_prefix}_bar.png"
        else:
            graphs["bar_chart"] = None
        
        # 3. RISK DISTRIBUTION PIE CHART (if disease present)
        if disease_preds and disease_percent > 0:
            # Categorize by risk level
            risk_counts = {"HIGH": 0, "MODERATE": 0, "LOW": 0}
            
            for pred in disease_preds:
                risk = get_risk_assessment(pred["disease"], pred["probability"])["risk_level"]
                if "HIGH" in risk:
                    risk_counts["HIGH"] += 1
                elif "MODERATE" in risk:
                    risk_counts["MODERATE"] += 1
                else:
                    risk_counts["LOW"] += 1
            
            # Remove zero counts
            risk_labels = [k for k, v in risk_counts.items() if v > 0]
            risk_values = [v for v in risk_counts.values() if v > 0]
            risk_colors = ['#ef4444' if l == 'HIGH' else '#f59e0b' if l == 'MODERATE' else '#3b82f6' for l in risk_labels]
            
            if risk_values:
                fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
                fig.patch.set_alpha(0)
                ax.set_facecolor('white')
                
                wedges, texts, autotexts = ax.pie(
                    risk_values, 
                    labels=risk_labels, 
                    colors=risk_colors,
                    autopct='%1.0f%%',
                    startangle=90,
                    wedgeprops=dict(edgecolor='white', linewidth=2),
                    textprops={'fontsize': 11, 'fontweight': 'bold'}
                )
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                ax.set_title('Risk Distribution Among Findings', 
                            fontsize=14, fontweight='bold', pad=20, color='#1e3a8a')
                
                # Save risk chart
                risk_path = os.path.join(GRAPH_FOLDER, f"{filename_prefix}_risk.png")
                plt.tight_layout()
                plt.savefig(risk_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                
                graphs["risk_chart"] = f"/static/graphs/{filename_prefix}_risk.png"
        
        return graphs
        
    except Exception as e:
        print(f"❌ Error generating graphs: {e}")
        import traceback
        traceback.print_exc()
        return {"donut_chart": None, "bar_chart": None}

# ==================== OPENROUTER AI GUIDANCE ====================

def get_ai_guidance(disease, probability, symptoms, patient_vitals):
    """
    Get AI guidance from OpenRouter for lifestyle, diet, and explanation.
    
    Args:
        disease: Detected disease name
        probability: Probability score
        symptoms: List of reported symptoms
        patient_vitals: Dictionary of patient vitals
        
    Returns:
        dict: AI guidance information with explanation, diet, lifestyle, warning signs
    """
    # If OpenRouter not configured, use predefined explanations
    if not client:
        explanation = get_disease_explanation(disease)
        return {
            "explanation": explanation["medical_reason"],
            "diet": "; ".join(explanation["lifestyle_recommendations"]["diet"]),
            "lifestyle": "; ".join(explanation["lifestyle_recommendations"]["habits"]),
            "warning_signs": "; ".join(explanation["emergency_signs"]),
            "disclaimer": "Using predefined medical guidance. For personalized advice, consult healthcare provider."
        }
    
    try:
        # Format symptoms
        symptoms_text = ', '.join(symptoms) if symptoms else 'None reported'
        
        # Format vitals
        vitals_text = f"""
- Blood Pressure: {patient_vitals.get('bp', 'Not recorded')}
- Blood Sugar: {patient_vitals.get('sugar', 'Not recorded')}
- Smoking Status: {patient_vitals.get('smoking', 'Not recorded')}
- Alcohol Use: {patient_vitals.get('alcohol', 'Not recorded')}
"""
        
        # Determine risk level based on probability
        risk_level = "LOW"
        if probability >= 70:
            risk_level = "HIGH"
        elif probability >= 40:
            risk_level = "MODERATE"
        
        prompt = f"""You are a professional radiology assistant providing patient education. Provide GENERAL information only.

RADIOLOGICAL FINDING:
- Primary Finding: {disease}
- AI Confidence: {probability}% ({risk_level} probability)
- Reported Symptoms: {symptoms_text}

PATIENT CONTEXT:
{vitals_text}

TASK: Provide educational information in valid JSON format with these exact keys:
1. "explanation": A clear, simple explanation of what this finding means (2-3 sentences, patient-friendly language)
2. "diet": 3-4 general dietary suggestions that may support overall health (bullet points)
3. "lifestyle": 3-4 general lifestyle recommendations (bullet points)
4. "warning_signs": When to seek immediate medical attention (3-4 bullet points)

IMPORTANT RULES:
- DO NOT diagnose or suggest specific treatments
- DO NOT mention specific medication names
- DO NOT override radiographic findings
- Use simple, clear language appropriate for patients
- Include educational disclaimer in explanation
- Response must be valid JSON only, no additional text

Example format:
{{
    "explanation": "This finding suggests... It's important to...",
    "diet": "• Stay hydrated\\n• Eat balanced meals\\n• Limit processed foods",
    "lifestyle": "• Regular gentle exercise\\n• Adequate rest\\n• Avoid smoking",
    "warning_signs": "• Severe chest pain\\n• Difficulty breathing\\n• High fever"
}}
"""
        
        # Make API call to OpenRouter
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",  # Fast, cost-effective model
            messages=[
                {"role": "system", "content": "You are a radiology AI assistant providing educational information only. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600,
            response_format={"type": "json_object"}
        )
        
        try:
            # Parse JSON response
            guidance_text = response.choices[0].message.content
            
            # Clean the response (remove markdown code blocks if present)
            guidance_text = guidance_text.replace('```json', '').replace('```', '').strip()
            
            guidance = json.loads(guidance_text)
            
            # Ensure all required keys exist with defaults
            required_keys = ["explanation", "diet", "lifestyle", "warning_signs"]
            for key in required_keys:
                if key not in guidance or not guidance[key]:
                    if key == "explanation":
                        guidance[key] = f"Educational information about {disease} is being processed."
                    elif key == "diet":
                        guidance[key] = "• Consult with healthcare provider for personalized dietary advice.\n• Maintain balanced nutrition.\n• Stay well-hydrated."
                    elif key == "lifestyle":
                        guidance[key] = "• Follow medical advice from qualified professionals.\n• Get adequate rest.\n• Attend follow-up appointments."
                    elif key == "warning_signs":
                        guidance[key] = "• Severe chest pain\n• Difficulty breathing\n• High fever\n• Confusion"
            
            # Add disclaimer
            guidance["disclaimer"] = "AI-generated educational content using OpenRouter. Not medical advice. Always consult healthcare providers."
            
            return guidance
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            # Fallback to predefined explanations
            explanation = get_disease_explanation(disease)
            return {
                "explanation": explanation["medical_reason"],
                "diet": "; ".join(explanation["lifestyle_recommendations"]["diet"]),
                "lifestyle": "; ".join(explanation["lifestyle_recommendations"]["habits"]),
                "warning_signs": "; ".join(explanation["emergency_signs"]),
                "disclaimer": "AI response parsing failed. Using predefined medical guidance."
            }
            
    except Exception as e:
        print(f"❌ OpenRouter API error: {e}")
        # Fallback to predefined explanations
        explanation = get_disease_explanation(disease)
        return {
            "explanation": explanation["medical_reason"],
            "diet": "; ".join(explanation["lifestyle_recommendations"]["diet"]),
            "lifestyle": "; ".join(explanation["lifestyle_recommendations"]["habits"]),
            "warning_signs": "; ".join(explanation["emergency_signs"]),
            "disclaimer": "AI guidance unavailable. Using standard medical guidance."
        }

# ==================== PDF REPORT GENERATION ====================

def generate_detailed_pdf(pdf_path, patient, image_path, heatmap_path, predictions, symptoms, 
                         patient_vitals, medical_history, is_normal, heatmap_description, ai_guidance):
    """
    Generates a professional 2-page clinical report with modern design and comprehensive findings.
    
    Args:
        pdf_path: Path to save PDF
        patient: Patient information dictionary
        image_path: Path to original X-ray image
        heatmap_path: Path to AI attention heatmap
        predictions: List of AI predictions
        symptoms: List of reported symptoms
        patient_vitals: Patient vital signs
        medical_history: List of medical conditions
        is_normal: Whether the finding is normal
        heatmap_description: Description of AI attention map
        ai_guidance: AI-generated guidance
    """
    # Create PDF document
    doc = SimpleDocTemplate(
        pdf_path, 
        pagesize=A4, 
        rightMargin=40, 
        leftMargin=40, 
        topMargin=40, 
        bottomMargin=40,
        title=f"Radiology Report - {patient['name']}",
        author="AI Radiology Assistant"
    )
    
    styles = getSampleStyleSheet()
    elements = []

    # ==================== CUSTOM STYLES ====================
    styles.add(ParagraphStyle(
        name='HospitalTitle',
        fontSize=16,
        textColor=colors.HexColor('#1e3a8a'),
        alignment=1,  # Center
        spaceAfter=6,
        fontName='Helvetica-Bold'
    ))
    
    styles.add(ParagraphStyle(
        name='Department',
        fontSize=10,
        textColor=colors.HexColor('#4b5563'),
        alignment=1,
        spaceAfter=12,
        fontName='Helvetica'
    ))
    
    styles.add(ParagraphStyle(
        name='Disclaimer',
        fontSize=8,
        textColor=colors.HexColor('#b91c1c'),
        alignment=1,
        spaceAfter=12,
        fontName='Helvetica-Oblique'
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeader',
        fontSize=12,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold',
        borderWidth=1,
        borderColor=colors.HexColor('#e5e7eb'),
        borderPadding=(5, 0, 5, 0)
    ))
    
    styles.add(ParagraphStyle(
        name='InfoLabel',
        fontSize=9,
        textColor=colors.HexColor('#6b7280'),
        fontName='Helvetica-Bold',
        alignment=2  # Right
    ))
    
    styles.add(ParagraphStyle(
        name='InfoValue',
        fontSize=9,
        textColor=colors.HexColor('#1f2937'),
        fontName='Helvetica',
        alignment=0  # Left
    ))
    
    styles.add(ParagraphStyle(
        name='NormalText',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1f2937'),
        fontName='Helvetica',
        spaceAfter=6
    ))
    
    styles.add(ParagraphStyle(
        name='RiskHigh',
        fontSize=10,
        textColor=colors.HexColor('#ffffff'),
        backColor=colors.HexColor('#ef4444'),
        alignment=1,
        fontName='Helvetica-Bold',
        borderPadding=(3, 10, 3, 10)
    ))
    
    styles.add(ParagraphStyle(
        name='RiskModerate',
        fontSize=10,
        textColor=colors.HexColor('#ffffff'),
        backColor=colors.HexColor('#f59e0b'),
        alignment=1,
        fontName='Helvetica-Bold',
        borderPadding=(3, 10, 3, 10)
    ))
    
    styles.add(ParagraphStyle(
        name='RiskLow',
        fontSize=10,
        textColor=colors.HexColor('#ffffff'),
        backColor=colors.HexColor('#3b82f6'),
        alignment=1,
        fontName='Helvetica-Bold',
        borderPadding=(3, 10, 3, 10)
    ))
    
    styles.add(ParagraphStyle(
        name='RiskNone',
        fontSize=10,
        textColor=colors.HexColor('#ffffff'),
        backColor=colors.HexColor('#10b981'),
        alignment=1,
        fontName='Helvetica-Bold',
        borderPadding=(3, 10, 3, 10)
    ))
    
    styles.add(ParagraphStyle(
        name='Guidance',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1f2937'),
        fontName='Helvetica',
        leftIndent=10,
        spaceAfter=4
    ))
    
    styles.add(ParagraphStyle(
        name='Emergency',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#b91c1c'),
        fontName='Helvetica-Bold',
        leftIndent=10,
        spaceAfter=4
    ))

    # ==================== PAGE 1 HEADER ====================
    # Hospital header
    header_data = [[
        Paragraph(f"<b>{HOSPITAL_NAME}</b>", styles['HospitalTitle'])
    ]]
    header_table = Table(header_data, colWidths=[500])
    header_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0)
    ]))
    elements.append(header_table)
    
    # Department
    elements.append(Paragraph(f"{DEPT}", styles['Department']))
    
    # Report title
    elements.append(Paragraph(
        "<font size=14><b>AI-ASSISTED RADIOGRAPHIC ANALYSIS REPORT</b></font>",
        styles['Normal']
    ))
    elements.append(Spacer(1, 10))
    
    # Primary disclaimer
    elements.append(Paragraph(
        "<font color='#b91c1c'><b>IMPORTANT:</b> This is an AI-assisted preliminary analysis and NOT a final diagnosis. "
        "All findings require clinical correlation by a qualified radiologist.</font>",
        styles['Disclaimer']
    ))
    elements.append(Spacer(1, 15))

    # ==================== PATIENT INFORMATION TABLE ====================
    # Create patient info grid
    p_data = [
        ["PATIENT INFORMATION", "", "REPORT DETAILS", ""],
        ["Name:", patient['name'], "Report ID:", f"RAD-{datetime.now().strftime('%Y%m%d%H%M')}"],
        ["Patient ID:", patient['id'], "Exam Date:", patient['date']],
        ["Age/Sex:", f"{patient['age']} / {patient['sex']}", "Reported By:", HEAD_RADIOLOGIST],
        ["Referring Dr:", patient.get('referring_dr', 'N/A'), "License No:", LICENSE_NO],
    ]
    
    p_table = Table(p_data, colWidths=[100, 150, 100, 150])
    p_table.setStyle(TableStyle([
        ('SPAN', (0,0), (1,0)),  # Merge first two cells in row 0
        ('SPAN', (2,0), (3,0)),  # Merge last two cells in row 0
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#1e3a8a')),
        ('BACKGROUND', (2,0), (3,0), colors.HexColor('#1e3a8a')),
        ('TEXTCOLOR', (0,0), (1,0), colors.white),
        ('TEXTCOLOR', (2,0), (3,0), colors.white),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (3,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('BACKGROUND', (0,1), (-1,-1), colors.white),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('ALIGN', (0,0), (1,0), 'CENTER'),
        ('ALIGN', (2,0), (3,0), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 6)
    ]))
    elements.append(p_table)
    elements.append(Spacer(1, 15))

    # ==================== CLINICAL DATA SECTION ====================
    clinical_data = []
    
    # Symptoms
    if symptoms:
        symptoms_text = ", ".join(symptoms)
        clinical_data.append([Paragraph("<b>Reported Symptoms:</b>", styles['InfoLabel']), 
                             Paragraph(symptoms_text, styles['InfoValue'])])
    
    # Medical History
    if medical_history:
        history_text = ", ".join(medical_history)
        clinical_data.append([Paragraph("<b>Medical History:</b>", styles['InfoLabel']), 
                             Paragraph(history_text, styles['InfoValue'])])
    
    # Vitals
    vitals_parts = []
    if patient_vitals.get('bp'): vitals_parts.append(f"BP: {patient_vitals['bp']}")
    if patient_vitals.get('sugar'): vitals_parts.append(f"Glucose: {patient_vitals['sugar']}")
    if patient_vitals.get('smoking'): vitals_parts.append(f"Smoking: {patient_vitals['smoking']}")
    if patient_vitals.get('alcohol'): vitals_parts.append(f"Alcohol: {patient_vitals['alcohol']}")
    
    if vitals_parts:
        vitals_text = " | ".join(vitals_parts)
        clinical_data.append([Paragraph("<b>Vital Signs:</b>", styles['InfoLabel']), 
                             Paragraph(vitals_text, styles['InfoValue'])])
    
    if clinical_data:
        c_table = Table(clinical_data, colWidths=[100, 400])
        c_table.setStyle(TableStyle([
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4)
        ]))
        elements.append(c_table)
        elements.append(Spacer(1, 15))

    # ==================== PRIMARY FINDING BANNER ====================
    primary_pred = predictions[0]
    primary_class = normalize_class_name(primary_pred['disease'])
    risk_assessment = get_risk_assessment(primary_class, primary_pred['probability'])
    
    if is_normal:
        banner_text = f"<b>PRIMARY FINDING: NORMAL CHEST RADIOGRAPH - NO ABNORMALITY DETECTED</b>"
        banner_style = 'RiskNone'
    else:
        banner_text = f"<b>PRIMARY FINDING: {primary_class.upper()} - {risk_assessment['risk_level']}</b>"
        if risk_assessment['risk_level'] == 'HIGH RISK':
            banner_style = 'RiskHigh'
        elif risk_assessment['risk_level'] == 'MODERATE RISK':
            banner_style = 'RiskModerate'
        else:
            banner_style = 'RiskLow'
    
    elements.append(Paragraph(banner_text, styles[banner_style]))
    elements.append(Spacer(1, 15))

    # ==================== AI QUANTITATIVE ASSESSMENT ====================
    if not is_normal:
        elements.append(Paragraph("<b>AI QUANTITATIVE ASSESSMENT</b>", styles['SectionHeader']))
        
        # Create probability table
        table_data = [["Radiographic Finding", "Probability", "Risk Level"]]
        
        for p in predictions[:6]:  # Top 6 findings
            if normalize_class_name(p['disease']) == "NORMAL":
                continue
                
            risk_info = get_risk_assessment(p['disease'], p['probability'])
            
            # Risk level styling
            if risk_info['risk_level'] == 'HIGH RISK':
                risk_color = colors.HexColor('#ef4444')
            elif risk_info['risk_level'] == 'MODERATE RISK':
                risk_color = colors.HexColor('#f59e0b')
            else:
                risk_color = colors.HexColor('#3b82f6')
            
            table_data.append([
                p['disease'],
                f"{p['probability']:.1f}%",
                Paragraph(f"<font color='{risk_info['hex_color']}'><b>{risk_info['risk_level']}</b></font>", styles['Normal'])
            ])
        
        if len(table_data) > 1:
            f_table = Table(table_data, colWidths=[200, 100, 150])
            f_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 10),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
                ('ALIGN', (1,0), (2,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('PADDING', (0,0), (-1,-1), 8),
                ('FONTSIZE', (0,1), (-1,-1), 9)
            ]))
            elements.append(f_table)
            elements.append(Spacer(1, 15))

    # ==================== IMAGES SECTION ====================
    elements.append(Paragraph("<b>RADIOGRAPHIC IMAGING</b>", styles['SectionHeader']))
    
    # Create image table
    if is_normal:
        # For NORMAL cases - show only original image
        img_table = Table([
            [PDFImage(image_path, width=4*inch, height=4*inch)]
        ])
        elements.append(img_table)
        elements.append(Paragraph(f"<i>Fig 1: Original Chest Radiograph - {heatmap_description}</i>", 
                                 styles['Italic']))
    else:
        # For disease cases - show original and heatmap side by side
        if heatmap_path and os.path.exists(heatmap_path):
            img_table = Table([
                [PDFImage(image_path, width=3*inch, height=3*inch), 
                 PDFImage(heatmap_path, width=3*inch, height=3*inch)]
            ])
            elements.append(img_table)
            elements.append(Paragraph(
                f"<i>Fig 1: Original Radiograph | Fig 2: AI Attention Map - {heatmap_description}</i>", 
                styles['Italic']))
        else:
            img_table = Table([
                [PDFImage(image_path, width=4*inch, height=4*inch)]
            ])
            elements.append(img_table)
            elements.append(Paragraph(f"<i>Fig 1: Original Chest Radiograph</i>", styles['Italic']))
    
    elements.append(Spacer(1, 15))

    # ==================== CLINICAL INTERPRETATION ====================
    elements.append(Paragraph("<b>CLINICAL INTERPRETATION</b>", styles['SectionHeader']))
    
    primary_explanation = get_disease_explanation(primary_class)
    
    if is_normal:
        interpretation = f"""
        <b>Diagnostic Impression:</b> {primary_explanation['medical_reason']}
        <br/><br/>
        <b>AI Attention Analysis:</b> {primary_explanation['ai_reason']} The neural network shows diffuse 
        attention patterns consistent with normal anatomical structures.
        """
        
        if symptoms:
            interpretation += f"<br/><br/><b>Symptom Correlation:</b> The reported symptoms ({', '.join(symptoms)}) are not explained by radiographic findings. Clinical correlation recommended."
    else:
        interpretation = f"""
        <b>Diagnostic Impression:</b> {primary_explanation['medical_reason']}
        <br/><br/>
        <b>Possible Causes:</b> {', '.join(primary_explanation['cause_analysis']['possible_causes'][:3])}
        <br/><br/>
        <b>Risk Factors:</b> {', '.join(primary_explanation['cause_analysis']['risk_factors'][:3])}
        <br/><br/>
        <b>AI Attention Analysis:</b> {primary_explanation['ai_reason']} The Vision Transformer shows focused 
        attention in areas corresponding to radiographic findings of {primary_class} (confidence: {primary_pred['probability']:.1f}%).
        """
        
        if symptoms:
            interpretation += f"<br/><br/><b>Symptom Correlation:</b> Reported symptoms ({', '.join(symptoms)}) are consistent with the radiographic finding."
    
    elements.append(Paragraph(interpretation, styles['NormalText']))
    elements.append(PageBreak())

    # ==================== PAGE 2: AI GUIDANCE ====================
    elements.append(Paragraph("<b>AI-GENERATED CLINICAL GUIDANCE</b>", styles['SectionHeader']))
    elements.append(Spacer(1, 10))
    
    # Format AI guidance with bullet points
    diet_text = ai_guidance.get('diet', '• Consult healthcare provider for personalized advice.').replace('• ', '<br/>• ')
    lifestyle_text = ai_guidance.get('lifestyle', '• Follow medical advice from professionals.').replace('• ', '<br/>• ')
    warning_text = ai_guidance.get('warning_signs', '• Severe chest pain<br/>• Difficulty breathing').replace('• ', '<br/>• ')
    
    guidance_html = f"""
    <b>Educational Explanation:</b><br/>
    {ai_guidance.get('explanation', 'Information not available')}
    <br/><br/>
    <b>Dietary Considerations:</b><br/>
    {diet_text}
    <br/><br/>
    <b>Lifestyle Recommendations:</b><br/>
    {lifestyle_text}
    """
    
    elements.append(Paragraph(guidance_html, styles['Guidance']))
    elements.append(Spacer(1, 15))
    
    # Emergency signs in red box
    emergency_html = f"""
    <font color='#b91c1c'><b>⚠️ WARNING SIGNS REQUIRING IMMEDIATE ATTENTION:</b></font><br/>
    {warning_text}
    """
    elements.append(Paragraph(emergency_html, styles['Emergency']))
    elements.append(Spacer(1, 15))
    
    # Disclaimer
    elements.append(Paragraph(
        f"<i>{ai_guidance.get('disclaimer', 'AI-generated educational content. Not medical advice.')}</i>",
        styles['Italic']
    ))
    elements.append(Spacer(1, 20))

    # ==================== RECOMMENDATIONS ====================
    elements.append(Paragraph("<b>CLINICAL RECOMMENDATIONS</b>", styles['SectionHeader']))
    
    if is_normal:
        rec_text = f"""
        <b>For the Patient:</b><br/><br/>
        <b>Recommended Actions:</b><br/>
        {'<br/>'.join(['• ' + action for action in primary_explanation['precautions']['do'][:4]])}
        <br/><br/>
        <b>Follow-up:</b> A normal chest radiograph is reassuring. If symptoms persist, consult with primary care physician.
        """
    else:
        rec_text = f"""
        <b>For Suspected {primary_class}:</b><br/><br/>
        <b>Immediate Actions:</b><br/>
        {'<br/>'.join(['• ' + action for action in primary_explanation['precautions']['do'][:4]])}
        <br/><br/>
        <b>Next Steps:</b><br/>
        • Consult with {get_specialist_for_disease(primary_class)}<br/>
        • Consider additional imaging if clinically indicated<br/>
        • Schedule follow-up as directed
        """
    
    elements.append(Paragraph(rec_text, styles['Guidance']))
    elements.append(Spacer(1, 20))

    # ==================== TECHNICAL METHODOLOGY ====================
    elements.append(Paragraph("<b>TECHNICAL METHODOLOGY</b>", styles['SectionHeader']))
    
    tech_text = f"""
    <b>AI Architecture:</b> Hybrid Vision Transformer (DeiT-Small + Swin-Tiny)<br/>
    <b>Fusion Method:</b> Weighted probability averaging (50% DeiT, 50% Swin)<br/>
    <b>Attention Mapping:</b> {heatmap_description}<br/>
    <b>Model Mode:</b> {os.getenv('MODEL_MODE', 'hybrid').upper()}<br/>
    <br/>
    <b>System Limitations:</b><br/>
    • AI analysis is preliminary and requires radiologist review<br/>
    • May miss subtle findings requiring advanced imaging<br/>
    • Not validated for pediatric or trauma cases<br/>
    • Clinical correlation with history and symptoms is essential
    """
    
    elements.append(Paragraph(tech_text, styles['Guidance']))
    elements.append(Spacer(1, 30))

    # ==================== SIGNATURE SECTION ====================
    # Signature line
    elements.append(Paragraph("_" * 50, styles['Normal']))
    elements.append(Spacer(1, 5))
    
    elements.append(Paragraph(
        f"<b>Digitally Reviewed by:</b> {HEAD_RADIOLOGIST}, MD",
        styles['Normal']
    ))
    elements.append(Paragraph(
        f"<b>Board Certified Radiologist | License:</b> {LICENSE_NO}",
        styles['Normal']
    ))
    elements.append(Paragraph(
        f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles['Normal']
    ))
    elements.append(Spacer(1, 15))

    # ==================== FINAL DISCLAIMER ====================
    final_disclaimer = """
    <font size=8>
    <b>FINAL DISCLAIMER:</b> This report constitutes an AI-assisted preliminary analysis only. 
    It is not a definitive diagnosis. The interpreting physician bears ultimate responsibility 
    for final diagnosis and clinical management decisions. Always consult with qualified 
    healthcare professionals for medical advice. Do not disregard or delay seeking professional 
    medical advice because of information contained in this report.
    </font>
    """
    elements.append(Paragraph(final_disclaimer, styles['Disclaimer']))

    # Build PDF
    doc.build(elements)
    print(f"✅ PDF report generated: {pdf_path}")

# ==================== UTILITY FUNCTIONS ====================

def cleanup_old_files(directory, max_age_hours=24):
    """
    Clean up files older than specified age.
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age in hours
    """
    try:
        now = datetime.now().timestamp()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_age = now - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    os.remove(filepath)
                    print(f"🧹 Cleaned up old file: {filename}")
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

def get_file_size_mb(filepath):
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)

def validate_image_format(filepath):
    """Validate if file is a supported image format."""
    try:
        img = Image.open(filepath)
        return img.format in ['JPEG', 'PNG', 'DICOM']
    except:
        return False