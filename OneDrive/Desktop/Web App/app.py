from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, g
import os
import json
from werkzeug.utils import secure_filename
# import rei  # Commented out due to import error
from datetime import datetime
import uuid
import shutil
from pdf_utils import PDFProcessor
import pdfplumber
from transformers import pipeline  # Make sure you have transformers installed
import time
import nltk
import random
import base64
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import logging

app = Flask(__name__, static_folder='static')
app.secret_key = os.urandom(24)

# Download required NLTK data
try:
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Ensure NLTK resources are downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.error(f"NLTK download error: {e}")

# Configuration
app.config['ALLOWED_PDF_EXTENSIONS'] = {'pdf'}
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['WHITEBOARD_IMAGES_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'whiteboard_images')

# Create necessary directories
def ensure_upload_directories():
    """Ensure all required upload directories exist."""
    directories = [
        app.config['UPLOAD_FOLDER'],
        app.config['WHITEBOARD_IMAGES_FOLDER'],
        os.path.join(app.config['UPLOAD_FOLDER'], 'pdf'),
        os.path.join(app.config['UPLOAD_FOLDER'], 'images')
    ]
    for directory in directories:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            raise

# Call this function when the app starts
ensure_upload_directories()

def allowed_file(filename):
    """
    Check if the file extension is allowed.
    
    Args:
        filename (str): Name of the file to check
    
    Returns:
        bool: True if file is allowed, False otherwise
    """
    if not filename or '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    # Determine allowed extensions based on current route
    if request.endpoint in ['upload_pdf', 'pdf_summary_post', 'process_pdf']:
        return ext in app.config['ALLOWED_PDF_EXTENSIONS']
    elif request.endpoint in ['upload_whiteboard_image', 'save_whiteboard']:
        return ext in app.config['ALLOWED_IMAGE_EXTENSIONS']
    
    return False

# Default theme settings
DEFAULT_THEME = {
    'color': '#0078d7',
    'font': 'Arial',
    'size': '16',
    'background_color': '#f4f4f9',
    'text_color': '#333333',
    'button_color': '#0078d7',
    'button_hover_color': '#005bb5',
    'button_text_color': '#ffffff',
    'card_background_color': '#ffffff',
    'border_color': '#e0e0e0'
}

def save_theme_data(theme_data):
    """Save theme data to a JSON file"""
    theme_file = os.path.join(app.root_path, 'static', 'theme.json')
    os.makedirs(os.path.dirname(theme_file), exist_ok=True)
    with open(theme_file, 'w') as f:
        json.dump(theme_data, f)

def load_theme_data():
    """Load theme data from JSON file"""
    theme_file = os.path.join(app.root_path, 'static', 'theme.json')
    if os.path.exists(theme_file):
        with open(theme_file, 'r') as f:
            return json.load(f)
    return {
        'color': '#0078d7',
        'background_color': '#f4f4f9',
        'text_color': '#333333',
        'button_color': '#0078d7',
        'button_hover_color': '#005bb5',
        'button_text_color': '#ffffff',
        'font': 'Arial'
    }

@app.context_processor
def inject_theme_data():
    """Inject theme data into all templates"""
    theme_data = session.get('theme_data') or load_theme_data()
    return {'theme_data': theme_data}

@app.route('/settings', methods=['GET'])
def settings():
    theme_data = session.get('theme_data', {
        'banner_color': '#FFB6C1',
        'background_color': '#FFF5F5',
        'button_color': '#FF69B4'
    })
    return render_template('settings.html', theme_data=theme_data)

@app.route('/save_theme', methods=['POST'])
def save_theme():
    # Cycle through predefined themes
    themes = [
        {
            'banner_color': '#FFB6C1',
            'background_color': '#FFF5F5',
            'button_color': '#FF69B4',
            'text_color': '#333'
        },
        {
            'banner_color': '#4169E1',
            'background_color': '#F0F8FF',
            'button_color': '#1E90FF',
            'text_color': '#000'
        },
        {
            'banner_color': '#2E8B57',
            'background_color': '#F0FFF0',
            'button_color': '#3CB371',
            'text_color': '#333'
        }
    ]

    # Get current theme or default to first theme
    current_theme = session.get('theme_data', themes[0])
    current_index = next((i for i, theme in enumerate(themes) if theme == current_theme), -1)
    
    # Cycle to next theme
    next_index = (current_index + 1) % len(themes)
    next_theme = themes[next_index]
    
    # Update session
    session['theme_data'] = next_theme
    
    return jsonify(next_theme)

@app.before_request
def before_request():
    # Set default theme if not present
    if 'theme_data' not in session:
        session['theme_data'] = {
            'banner_color': '#FFB6C1',
            'background_color': '#FFF5F5',
            'button_color': '#FF69B4'
        }
    g.theme_data = session['theme_data']

def clean_text(text):
    """
    Clean and preprocess text for better processing.
    
    Args:
        text (str): Input text to clean
    
    Returns:
        str: Cleaned and normalized text
    """
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

def extract_meaningful_sentences(text, min_words=6, max_words=50):
    """
    Extract meaningful sentences based on word count.
    
    Args:
        text (str): Input text
        min_words (int): Minimum number of words in a sentence
        max_words (int): Maximum number of words in a sentence
    
    Returns:
        list: List of meaningful sentences
    """
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    
    meaningful_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        if min_words <= len(filtered_words) <= max_words:
            meaningful_sentences.append(sentence)
    
    return meaningful_sentences

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('pdf_processing.log'),
                        logging.StreamHandler()
                    ])

def process_pdf(file_path, num_points=10, num_questions=5):
    """
    Process PDF file and extract key information with comprehensive error handling.
    
    Args:
        file_path (str): Path to the PDF file
        num_points (int): Number of summary points to generate
        num_questions (int): Number of questions to generate
    
    Returns:
        dict: Processed PDF information
    """
    # Validate inputs
    if not isinstance(num_points, int) or num_points <= 0:
        num_points = 10
    if not isinstance(num_questions, int) or num_questions <= 0:
        num_questions = 5

    try:
        # Validate file exists and is readable
        if not os.path.exists(file_path):
            logging.error(f"PDF file not found: {file_path}")
            return {
                'success': False,
                'error': 'PDF file not found'
            }
        
        # Check file size (limit to 50MB)
        max_file_size = 50 * 1024 * 1024  # 50 MB
        file_size = os.path.getsize(file_path)
        if file_size > max_file_size:
            logging.warning(f"PDF file too large: {file_size} bytes")
            return {
                'success': False,
                'error': 'PDF file is too large (max 50MB)'
            }
        
        # Open the PDF file
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                full_text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    full_text += page_text + "\n"
        except Exception as read_error:
            logging.error(f"Error reading PDF: {read_error}")
            return {
                'success': False,
                'error': f'Error reading PDF: {str(read_error)}'
            }
        
        # Validate extracted text
        if not full_text.strip():
            logging.warning("No text extracted from PDF")
            return {
                'success': False,
                'error': 'No text could be extracted from the PDF'
            }
        
        # Clean the extracted text
        try:
            cleaned_text = clean_text(full_text)
        except Exception as clean_error:
            logging.error(f"Error cleaning text: {clean_error}")
            cleaned_text = full_text
        
        # Extract meaningful sentences
        try:
            sentences = extract_meaningful_sentences(full_text)
        except Exception as sentence_error:
            logging.error(f"Error extracting sentences: {sentence_error}")
            sentences = full_text.split('.')[:num_points]
        
        # Validate sentences
        if not sentences:
            logging.warning("Could not extract meaningful sentences")
            return {
                'success': False,
                'error': 'Could not extract meaningful sentences from the PDF'
            }
        
        # Generate summary points
        summary_points = sentences[:num_points]
        
        # Generate questions 
        questions = [
            f"What is the main idea behind: {sentence}?"
            for sentence in sentences[:num_questions]
        ]
        
        # Log successful processing
        logging.info(f"Successfully processed PDF: {os.path.basename(file_path)}")
        
        return {
            'success': True,
            'text_length': len(full_text),
            'summary_points': summary_points,
            'questions': questions,
            'file_name': os.path.basename(file_path)
        }
    
    except Exception as unexpected_error:
        logging.error(f"Unexpected PDF processing error: {unexpected_error}", exc_info=True)
        return {
            'success': False,
            'error': str(unexpected_error)
        }
    finally:
        # Attempt to remove the temporary file
        try:
            os.remove(file_path)
        except Exception as cleanup_error:
            logging.warning(f"Could not delete temporary PDF file: {cleanup_error}")

@app.route('/')
def index():
    theme_data = session.get('theme_data', {
        'banner_color': '#FFB6C1',
        'background_color': '#FFF5F5',
        'button_color': '#FF69B4'
    })
    return render_template('index.html', theme_data=theme_data)

@app.route('/pdf_summary', methods=['GET'])
def pdf_summary():
    """
    Render the PDF summary page.
    """
    return render_template('pdf_summary.html')

@app.route('/process_pdf', methods=['POST'])
def pdf_summary_post():
    """
    Handle PDF summary generation with comprehensive error handling.
    Supports multiple input formats and provides detailed error responses.
    """
    try:
        # Check for file in multiple possible locations
        file = None
        if 'file' in request.files:
            file = request.files['file']
        elif 'pdf' in request.files:
            file = request.files['pdf']
        
        if file is None:
            return jsonify({
                'error': 'No PDF file found in request',
                'details': 'Please upload a valid PDF file',
                'status': 400
            }), 400
        
        # Validate filename
        if file.filename == '':
            return jsonify({
                'error': 'No selected file',
                'details': 'The uploaded file has no filename',
                'status': 400
            }), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'details': 'Only PDF files are allowed',
                'status': 400
            }), 400
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'pdf', filename)
        
        # Ensure upload directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)
        
        # Process PDF
        result = process_pdf(filepath)
        
        # Clean up temporary file after processing
        try:
            os.remove(filepath)
        except Exception as cleanup_error:
            logging.warning(f"Could not delete temporary PDF file: {cleanup_error}")
        
        if result['success']:
            return jsonify({
                'summary_points': result['summary_points'],
                'questions': result['questions'],
                'file_name': result['file_name'],
                'text_length': result.get('text_length', 0),
                'status': 200
            }), 200
        else:
            return jsonify({
                'error': 'PDF Processing Failed',
                'details': result.get('error', 'Unknown processing error'),
                'status': 500
            }), 500
    
    except json.JSONDecodeError:
        return jsonify({
            'error': 'Invalid JSON',
            'details': 'The request body is not a valid JSON',
            'status': 400
        }), 400
    
    except Exception as e:
        logging.error(f"Unexpected error in PDF summary: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'details': str(e),
            'status': 500
        }), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Handle PDF file upload with proper error handling."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'pdf', filename)
        
        try:
            file.save(filepath)
        except Exception as e:
            return jsonify({'error': f'Error saving file: {str(e)}'}), 500
            
        return jsonify({'success': True, 'filepath': filepath})
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/upload_whiteboard_image', methods=['POST'])
def upload_whiteboard_image():
    """Handle whiteboard image upload with proper error handling."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg, gif'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['WHITEBOARD_IMAGES_FOLDER'], filename)
        
        try:
            file.save(filepath)
        except Exception as e:
            return jsonify({'error': f'Error saving image: {str(e)}'}), 500
            
        return jsonify({'success': True, 'filepath': filepath})
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/flashcards', methods=['GET', 'POST'])
def flashcards_view():
    theme_data = session.get('theme_data', {
        'banner_color': '#FFB6C1',
        'background_color': '#FFF5F5',
        'button_color': '#FF69B4'
    })
    if 'flashcards' not in session:
        session['flashcards'] = []
        
    if request.method == 'POST':
        front = request.form.get('front')
        back = request.form.get('back')
        
        if front and back:  # Only add if both fields are filled
            session['flashcards'] = session.get('flashcards', []) + [{
                'front': front,
                'back': back
            }]
            
            # Save the updated flashcards list to session
            session.modified = True
            
            return jsonify({
                'status': 'success',
                'message': 'Flashcard created successfully',
                'flashcard': {'front': front, 'back': back}
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Both front and back are required'
            }), 400
            
    return render_template('flashcards.html', 
                         flashcards=session.get('flashcards', []), theme_data=theme_data)

@app.route('/delete_flashcard/<int:index>', methods=['POST'])
def delete_flashcard(index):
    if 'flashcards' in session and 0 <= index < len(session['flashcards']):
        session['flashcards'].pop(index)
        session.modified = True
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'}), 404

@app.route('/study_planner', methods=['GET', 'POST'])
def study_planner_view():
    theme_data = session.get('theme_data', {
        'banner_color': '#FFB6C1',
        'background_color': '#FFF5F5',
        'button_color': '#FF69B4'
    })
    if request.method == 'POST':
        session['study_plan'] = request.form.get('study_plan')
    return render_template('study_planner.html', theme_data=theme_data)

@app.route('/advanced_notes')
def advanced_notes():
    return redirect(url_for('whiteboard'))

@app.route('/whiteboard')
def whiteboard():
    theme_data = session.get('theme_data', {
        'banner_color': '#FFB6C1',
        'background_color': '#FFF5F5',
        'button_color': '#FF69B4'
    })
    """Render the whiteboard page."""
    return render_template('whiteboard.html', theme_data=theme_data)

@app.route('/whiteboard/save', methods=['POST'])
def save_whiteboard():
    """Save the whiteboard state."""
    try:
        data = request.json
        if not data or 'imageData' not in data:
            return jsonify({'success': False, 'error': 'No data provided'})
            
        # Generate a unique filename
        filename = f'whiteboard_{int(time.time())}.png'
        filepath = os.path.join(app.config['WHITEBOARD_IMAGES_FOLDER'], filename)
        
        # Save the image data
        image_data = data['imageData'].split(',')[1]  # Remove the data URL prefix
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_data))
            
        return jsonify({
            'success': True,
            'url': url_for('static', filename=f'uploads/whiteboard_images/{filename}')
        })
    except Exception as e:
        print(f"Error saving whiteboard: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)