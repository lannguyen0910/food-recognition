# Food Recognition & Nutrition Dashboard

AI-powered food recognition and nutrition tracking application using YOLOv8 for food detection and Claude AI for personalized health assessments.

## Features

- **Food Detection**: Real-time food recognition using YOLOv8
- **Nutrition Analysis**: Automatic calorie and macronutrient estimation
- **AI Health Coach**: Personalized health scores and recommendations powered by Claude AI
- **User Profiles**: Track health metrics (BMI, age, activity level, health goals)
- **Detection History**: View and analyze your meal history
- **Multi-unit Support**: Weight (kg/lbs) and height (cm/ft-in) conversion
- **Secure Authentication**: bcrypt password hashing, CSRF protection, rate limiting
- **PostgreSQL Database**: Production-ready data persistence

## Tech Stack

- **Backend**: Flask, Python 3.12+
- **AI/ML**: YOLOv8 (Ultralytics), Claude AI (Anthropic)
- **Database**: PostgreSQL (production), SQLite (local dev)
- **Security**: bcrypt, Flask-WTF (CSRF), Flask-Limiter
- **Frontend**: Vanilla JavaScript, Modern CSS

## Security Features

- bcrypt password hashing with salt
- CSRF protection on all forms
- Rate limiting to prevent abuse:
  - Login: 10 attempts/minute
  - Signup: 5/hour
  - Food Detection: 30/hour
  - Profile Updates: 20/hour
- Input validation and sanitization
- Secure session management

## Installation

### Prerequisites

- Python 3.12+
- PostgreSQL database
- Anthropic API key (for Claude AI health assessments)

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd food-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file with:
```bash
# Anthropic Claude API
ANTHROPIC_API_KEY=your_api_key_here

# Database (REQUIRED - PostgreSQL only)
DATABASE_URL=postgresql://username:password@host:port/database

# Flask Secret Key
SECRET_KEY=your_secret_key_here
```

4. Initialize the database:
The database will be automatically initialized on first run.

5. Download YOLO weights:
Place `yolov8s.pt` in the `weights/` directory, or the app will download `yolov8n.pt` automatically.

## Usage

### Run the server:
```bash
python app_dashboard.py --port 5001
```

### Access the application:
Open your browser to `http://localhost:5001`

### Command-line options:
```bash
python app_dashboard.py --help
  --host HOST    Host address (default: localhost)
  --port PORT    Port number (default: 5001)
  --debug        Enable debug mode
```

## Project Structure

```
food-recognition/
├── app_dashboard.py          # Main Flask application
├── backend/
│   ├── database.py           # PostgreSQL database functions
│   ├── claude_health_assessment.py  # Claude AI integration
│   ├── food_calories_estimation.py  # Nutrition calculation
│   └── yolo_detect.py        # YOLO food detection
├── static/
│   ├── css/                  # Stylesheets
│   ├── js/                   # JavaScript
│   ├── uploads/              # Uploaded images
│   └── detections/           # Detection results
├── templates/                # HTML templates
├── weights/                  # YOLO model weights
├── data/                     # SQLite database (if using SQLite)
└── requirements.txt          # Python dependencies
```

## Database Schema

### Users Table
- id, email, password_hash, google_id
- name, profile_picture
- weight_kg, height_cm, age, gender
- activity_level, health_goals, dietary_restrictions
- is_guest, created_at, updated_at

### Detection History Table
- id, user_id, timestamp
- image_path, detection_image_path
- detected_foods (JSONB)
- nutrition_data (JSONB)
- health_score, total_calories, total_protein, total_carbs, total_fat

## API Endpoints

### Authentication
- `POST /api/auth/signup` - Create account
- `POST /api/auth/login` - Login
- `POST /api/auth/guest` - Guest login
- `POST /api/auth/logout` - Logout

### User Profile
- `GET /api/user/profile` - Get user profile
- `PUT /api/user/profile` - Update profile

### Food Detection
- `POST /api/detect-food` - Upload image and detect food
- `GET /api/history` - Get detection history
- `GET /api/stats` - Get user statistics

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- Claude AI by Anthropic
- Food detection model training data from various open-source datasets
