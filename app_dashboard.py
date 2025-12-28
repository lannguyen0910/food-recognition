"""
Food Nutrition CV Dashboard Application
Uses custom YOLO model + LLaMA for nutrition insights
"""

import os
import cv2
import numpy as np
import torch
import secrets
from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect, generate_csrf
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import Llama nutrition module (optional)
try:
    from backend.llama_nutrition import analyze_food_nutrition
    LLAMA_AVAILABLE = True
    print("✓ Llama nutrition module loaded")
except ImportError as e:
    LLAMA_AVAILABLE = False
    print(f"⚠ Llama nutrition not available: {e}")
    # Fallback: use basic nutrition estimation
    from backend.food_calories_estimation import get_nutrition_data as analyze_food_nutrition

from backend.claude_health_assessment import get_health_score_from_claude
from backend.database import (
    add_detection, get_recent_detections, get_detection_by_id, get_stats,
    create_user, get_user_by_email, get_user_by_id, get_user_by_google_id,
    update_user_profile, verify_password
)
from backend.auth import get_current_user_id, verify_google_token

# Import segmentation modules
try:
    from theseus.apis.inference.segmentize import SegmentationPipeline
    from theseus.opt import Opts, InferenceArguments
    SEGMENTATION_AVAILABLE = True
    print("✓ Segmentation modules loaded")
except ImportError as e:
    SEGMENTATION_AVAILABLE = False
    print(f"⚠ Segmentation not available: {e}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['DETECTION_FOLDER'] = './static/detections'
app.config['SEGMENTATION_FOLDER'] = './static/segmentations'
app.config['MAX_CONTENT_LENGTH'] = 60 * 1024 * 1024  # 60MB max file size
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['WTF_CSRF_TIME_LIMIT'] = None  # CSRF tokens don't expire

# Initialize CSRF Protection
csrf = CSRFProtect(app)

# Initialize Rate Limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per hour"],  # Default rate limit
    storage_uri="memory://"  # Use memory storage (switch to Redis in production)
)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTION_FOLDER'], exist_ok=True)
os.makedirs(app.config['SEGMENTATION_FOLDER'], exist_ok=True)

# Model cache to store loaded models
model_cache = {}

def get_yolo_model(model_type='yolov8s'):
    """
    Get or load YOLO model based on type.

    Args:
        model_type: Either 'yolov8s' (weights) or 'yolov8n' (fallback)

    Returns:
        Loaded YOLO model
    """
    if model_type in model_cache:
        return model_cache[model_type]

    try:
        if model_type == 'yolov8s':
            model_path = os.getenv('YOLO_MODEL_PATH', './weights/yolov8s.pt')
            if os.path.exists(model_path):
                print(f"Loading YOLOv8s from {model_path}")
                model = YOLO(model_path)
                print("✓ YOLOv8s food detection model loaded")
            else:
                print(f"YOLOv8s not found at {model_path}, falling back to YOLOv8n")
                model = YOLO('yolov8n.pt')
        else:  # yolov8n
            print("Loading YOLOv8n fallback model")
            model = YOLO('yolov8n.pt')
            print("✓ YOLOv8n model loaded")

        model_cache[model_type] = model
        return model
    except Exception as e:
        print(f"Error loading {model_type}: {e}")
        print("Falling back to YOLOv8n")
        fallback = YOLO('yolov8n.pt')
        model_cache[model_type] = fallback
        return fallback

# Pre-load the default YOLOv8s model
print("=" * 60)
print("Initializing YOLO models...")
print("=" * 60)
yolo_model = get_yolo_model('yolov8s')


def run_segmentation(image_path):
    """
    Run semantic segmentation on the image.

    Args:
        image_path: Path to the input image

    Returns:
        Tuple of (overlay_path, mask_path) or (None, None) if segmentation fails
    """
    if not SEGMENTATION_AVAILABLE:
        print("Segmentation modules not available")
        return None, None

    try:
        # Create segmentation pipeline
        seg_args = InferenceArguments(key="segmentation")
        opts = Opts(seg_args).parse_args()
        seg_pipeline = SegmentationPipeline(opts, image_path)

        # Run segmentation
        output_path = seg_pipeline.inference()

        # The segmentation saves files to timestamped folders, extract the paths
        # output_path is the mask path, we also have overlay in a parallel directory
        mask_dir = os.path.dirname(output_path)
        overlay_dir = mask_dir.replace('/masks', '/overlays')
        filename = os.path.basename(output_path)
        overlay_path = os.path.join(overlay_dir, filename)

        # Copy to our static folder for easy serving
        import shutil
        dest_overlay = os.path.join(app.config['SEGMENTATION_FOLDER'], 'overlay_' + filename)
        dest_mask = os.path.join(app.config['SEGMENTATION_FOLDER'], 'mask_' + filename)

        if os.path.exists(overlay_path):
            shutil.copy(overlay_path, dest_overlay)
        if os.path.exists(output_path):
            shutil.copy(output_path, dest_mask)

        return dest_overlay, dest_mask

    except Exception as e:
        print(f"Segmentation error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


@app.route('/')
def index():
    """Render the main dashboard."""
    # Check if user is logged in or is guest
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('dashboard.html')


@app.route('/login')
def login_page():
    """Render the login/signup page."""
    # If already logged in, redirect to dashboard
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/api/csrf-token', methods=['GET'])
def get_csrf_token():
    """Generate and return a CSRF token for JavaScript requests."""
    token = generate_csrf()
    return jsonify({'csrf_token': token})


@app.route('/api/detect-food', methods=['POST'])
@limiter.limit("30 per hour")  # Limit food detection to prevent abuse
@csrf.exempt  # Exempt file uploads from CSRF (protected by rate limiting)
def detect_food():
    """
    Detect food items from uploaded image and generate nutrition insights.

    Returns:
        JSON response with detection results and nutrition data
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        # Get parameters
        confidence = float(request.form.get('confidence', 50)) / 100
        model_type = request.form.get('model', 'yolov8s')  # Default to yolov8s
        enable_segmentation = request.form.get('segmentation', 'false').lower() == 'true'

        # Normalize model type (backwards compatibility)
        if model_type == 'best':
            model_type = 'yolov8s'

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'success': False, 'error': 'Failed to load image'}), 400

        # Get the selected YOLO model
        selected_model = get_yolo_model(model_type)

        # Run YOLO detection
        results = selected_model.predict(
            source=image,
            conf=confidence,
            save=False,
            verbose=False
        )

        # Extract detected food items
        detected_foods = []
        result = results[0]

        # Draw bounding boxes
        annotated_image = result.plot()

        # Extract food names
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls]

            detected_foods.append({
                'name': class_name,
                'confidence': round(conf * 100, 2)
            })

        # Save annotated image
        detection_filename = f"detection_{filename}"
        detection_path = os.path.join(app.config['DETECTION_FOLDER'], detection_filename)
        cv2.imwrite(detection_path, annotated_image)

        # If no food items detected, return early without saving to database
        if len(detected_foods) == 0:
            print("No food items detected - skipping database save and stats")
            return jsonify({
                'success': True,
                'detection_image': f"/static/detections/{detection_filename}",
                'detected_foods': [],
                'nutrition_data': {
                    'foods': [],
                    'total_calories': 0,
                    'total_protein': 0,
                    'total_carbs': 0,
                    'total_fat': 0,
                    'health_score': 0,
                    'balance_assessment': 'No food items detected in the image',
                    'recommendations': []
                }
            })

        # Run segmentation if requested
        segmentation_overlay = None
        segmentation_mask = None
        if enable_segmentation and SEGMENTATION_AVAILABLE:
            print("Running semantic segmentation...")
            overlay_path, mask_path = run_segmentation(filepath)
            if overlay_path:
                segmentation_overlay = overlay_path.replace('./static', '/static')
            if mask_path:
                segmentation_mask = mask_path.replace('./static', '/static')
            print(f"✓ Segmentation completed")
        elif enable_segmentation and not SEGMENTATION_AVAILABLE:
            print("⚠ Segmentation requested but not available")

        # Get unique food names
        unique_foods = list(set([food['name'] for food in detected_foods]))

        # Generate nutrition insights with LLaMA
        print(f"Generating nutrition insights for: {unique_foods}")

        use_api = os.getenv('USE_LLAMA_API', 'false').lower() == 'true'
        nutrition_result = analyze_food_nutrition(unique_foods, use_api=use_api)

        if not nutrition_result['success']:
            print(f"LLaMA error: {nutrition_result.get('error')}")
            # Use fallback nutrition data
            nutrition_data = generate_basic_nutrition(unique_foods)
        else:
            nutrition_data = nutrition_result['data']

        # Get health score from Claude AI (doctor's perspective)
        print("Getting health assessment from Claude AI...")
        try:
            # Get user profile for personalized health scoring
            user_id = get_current_user_id()
            user_profile = get_user_by_id(user_id) if user_id else None

            claude_assessment = get_health_score_from_claude(unique_foods, nutrition_data, user_profile)
            # Update ONLY the health score from Claude
            nutrition_data['health_score'] = claude_assessment['health_score']
            print(f"✓ Claude health score: {claude_assessment['health_score']}/100")
        except Exception as claude_error:
            print(f"Claude assessment error: {claude_error}")
            # Keep existing health score from LLaMA or fallback

        # Save to database
        try:
            user_id = get_current_user_id()
            detection_id = add_detection(
                image_path=f"/static/uploads/{filename}",
                detection_image_path=f"/static/detections/{detection_filename}",
                detected_foods=detected_foods,
                nutrition_data=nutrition_data,
                user_id=user_id
            )
            print(f"✓ Detection saved to database with ID: {detection_id}")
        except Exception as db_error:
            print(f"Warning: Failed to save to database: {db_error}")

        # Build response
        response_data = {
            'success': True,
            'detection_image': f"/static/detections/{detection_filename}",
            'detected_foods': detected_foods,
            'nutrition_data': nutrition_data
        }

        # Add segmentation results if available
        if segmentation_overlay:
            response_data['segmentation_overlay'] = segmentation_overlay
        if segmentation_mask:
            response_data['segmentation_mask'] = segmentation_mask

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in detect_food: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def generate_basic_nutrition(food_items):
    """
    Generate basic nutrition estimates when LLaMA is unavailable.

    Args:
        food_items: List of food names

    Returns:
        Dictionary with basic nutrition data
    """
    # Very basic estimates
    basic_cal_per_item = 150
    basic_protein = 8
    basic_carbs = 20
    basic_fat = 5

    foods = []
    for item in food_items:
        foods.append({
            'name': item,
            'calories': basic_cal_per_item,
            'protein': basic_protein,
            'carbs': basic_carbs,
            'fat': basic_fat,
            'vitamins': ['Various'],
            'minerals': ['Various'],
            'benefits': 'Consult a nutritionist for accurate information'
        })

    total_items = len(food_items)

    return {
        'foods': foods,
        'total_calories': basic_cal_per_item * total_items,
        'total_protein': basic_protein * total_items,
        'total_carbs': basic_carbs * total_items,
        'total_fat': basic_fat * total_items,
        'health_score': 50,
        'balance_assessment': 'Basic estimate. LLaMA nutrition analysis unavailable.',
        'recommendations': [
            'Verify nutrition with a professional',
            'Maintain balanced diet',
            'Consider portion sizes'
        ]
    }


# Redirect old routes to dashboard
@app.route('/modern')
def modern_redirect():
    return redirect('/')


@app.route('/classic')
def classic():
    return render_template('upload-file.html')


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get recent detection history."""
    try:
        limit = request.args.get('limit', 10, type=int)
        user_id = get_current_user_id()
        history = get_recent_detections(limit=limit, user_id=user_id)

        return jsonify({
            'success': True,
            'history': history
        })
    except Exception as e:
        print(f"Error in get_history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/history/<int:detection_id>', methods=['GET'])
def get_history_item(detection_id):
    """Get a specific detection by ID."""
    try:
        detection = get_detection_by_id(detection_id)

        if not detection:
            return jsonify({
                'success': False,
                'error': 'Detection not found'
            }), 404

        return jsonify({
            'success': True,
            'detection': detection
        })
    except Exception as e:
        print(f"Error in get_history_item: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get overall statistics."""
    try:
        user_id = get_current_user_id()
        stats = get_stats(user_id=user_id)

        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        print(f"Error in get_statistics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# Authentication Routes
# ============================================================================

@app.route('/api/auth/signup', methods=['POST'])
@limiter.limit("5 per hour")  # Limit signups to prevent abuse
def signup():
    """User signup endpoint."""
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        name = data.get('name', '').strip()

        # Input validation
        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password required'}), 400

        # Email validation (basic)
        if '@' not in email or '.' not in email.split('@')[1]:
            return jsonify({'success': False, 'error': 'Invalid email format'}), 400

        # Password strength validation
        if len(password) < 6:
            return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400

        # Name validation
        if name and len(name) > 100:
            return jsonify({'success': False, 'error': 'Name too long'}), 400

        # Check if user already exists
        existing_user = get_user_by_email(email)
        if existing_user:
            return jsonify({'success': False, 'error': 'Email already registered'}), 400

        # Create user
        user_id = create_user(email=email, password=password, name=name)
        if not user_id:
            return jsonify({'success': False, 'error': 'Failed to create user'}), 500

        # Log in the user
        session['user_id'] = user_id
        session['user_email'] = email
        session['user_name'] = name

        return jsonify({
            'success': True,
            'user': {'id': user_id, 'email': email, 'name': name}
        })

    except Exception as e:
        print(f"Error in signup: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per minute")  # Prevent brute force attacks
def login():
    """User login endpoint."""
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '')

        # Input validation
        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password required'}), 400

        # Basic email format validation
        if '@' not in email:
            return jsonify({'success': False, 'error': 'Invalid email format'}), 400

        # Verify credentials
        user = verify_password(email, password)
        if not user:
            return jsonify({'success': False, 'error': 'Invalid email or password'}), 401

        # Log in the user
        session['user_id'] = user['id']
        session['user_email'] = user['email']
        session['user_name'] = user['name']

        return jsonify({
            'success': True,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'name': user['name'],
                'profile_picture': user.get('profile_picture')
            }
        })

    except Exception as e:
        print(f"Error in login: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@app.route('/api/auth/google', methods=['POST'])
def google_auth():
    """Google OAuth login endpoint."""
    try:
        data = request.get_json()
        id_token = data.get('id_token')

        if not id_token:
            return jsonify({'success': False, 'error': 'ID token required'}), 400

        # Verify Google token
        google_user = verify_google_token(id_token)
        if not google_user:
            return jsonify({'success': False, 'error': 'Invalid Google token'}), 401

        # Check if user exists
        user = get_user_by_google_id(google_user['google_id'])

        if not user:
            # Create new user
            user_id = create_user(
                email=google_user['email'],
                google_id=google_user['google_id'],
                name=google_user['name'],
                profile_picture=google_user.get('picture')
            )
            if not user_id:
                return jsonify({'success': False, 'error': 'Failed to create user'}), 500

            user = get_user_by_id(user_id)

        # Log in the user
        session['user_id'] = user['id']
        session['user_email'] = user['email']
        session['user_name'] = user['name']

        return jsonify({
            'success': True,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'name': user['name'],
                'profile_picture': user.get('profile_picture')
            }
        })

    except Exception as e:
        print(f"Error in Google auth: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auth/guest', methods=['POST'])
@limiter.limit("20 per hour")  # Limit guest account creation
def guest_login():
    """Continue as guest endpoint."""
    try:
        # Create a guest user
        guest_name = f"Guest_{secrets.token_hex(4)}"
        user_id = create_user(name=guest_name, is_guest=True)

        if not user_id:
            return jsonify({'success': False, 'error': 'Failed to create guest user'}), 500

        # Log in the guest user
        session['user_id'] = user_id
        session['user_name'] = guest_name
        session['is_guest'] = True

        return jsonify({
            'success': True,
            'user': {'id': user_id, 'name': guest_name, 'is_guest': True}
        })

    except Exception as e:
        print(f"Error in guest login: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """User logout endpoint."""
    session.clear()
    return jsonify({'success': True})


@app.route('/api/user/profile', methods=['GET'])
def get_profile():
    """Get current user profile."""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401

        user = get_user_by_id(user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        # Remove sensitive data
        user.pop('password_hash', None)

        return jsonify({'success': True, 'user': user})

    except Exception as e:
        print(f"Error getting profile: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/user/profile', methods=['PUT'])
@limiter.limit("20 per hour")  # Limit profile updates
def update_profile():
    """Update user profile."""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401

        data = request.get_json()

        # Input validation for profile fields
        if 'name' in data:
            name = data['name'].strip() if isinstance(data['name'], str) else ''
            if len(name) > 100:
                return jsonify({'success': False, 'error': 'Name too long'}), 400
            data['name'] = name

        if 'weight_kg' in data:
            try:
                weight = float(data['weight_kg'])
                if weight < 20 or weight > 500:
                    return jsonify({'success': False, 'error': 'Weight must be between 20-500 kg'}), 400
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': 'Invalid weight value'}), 400

        if 'height_cm' in data:
            try:
                height = float(data['height_cm'])
                if height < 50 or height > 300:
                    return jsonify({'success': False, 'error': 'Height must be between 50-300 cm'}), 400
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': 'Invalid height value'}), 400

        if 'age' in data:
            try:
                age = int(data['age'])
                if age < 1 or age > 150:
                    return jsonify({'success': False, 'error': 'Age must be between 1-150'}), 400
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': 'Invalid age value'}), 400

        # Update profile
        success = update_user_profile(user_id, **data)
        if not success:
            return jsonify({'success': False, 'error': 'Failed to update profile'}), 500

        # Get updated user
        user = get_user_by_id(user_id)
        user.pop('password_hash', None)

        # Update session if name changed
        if 'name' in data:
            session['user_name'] = data['name']

        return jsonify({'success': True, 'user': user})

    except Exception as e:
        print(f"Error updating profile: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Food Nutrition CV Dashboard')
    parser.add_argument('--host', type=str, default='localhost', help='Host address')
    parser.add_argument('--port', type=int, default=5001, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    print("=" * 60)
    print("Food Nutrition CV Dashboard")
    print("=" * 60)
    print(f"Models: YOLOv8s (Weights) + YOLOv8n (Fallback)")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )
