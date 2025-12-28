"""
Database module for food detection history
PostgreSQL only - no SQLite support
"""
import json
from datetime import datetime
import os
import bcrypt
import secrets
import psycopg2
from psycopg2.extras import RealDictCursor

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required. Please set it in your .env file.")


def get_connection():
    """Get PostgreSQL database connection."""
    return psycopg2.connect(DATABASE_URL)


def init_database():
    """Initialize the database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE,
            password_hash TEXT,
            google_id VARCHAR(255) UNIQUE,
            name VARCHAR(255),
            profile_picture TEXT,
            weight_kg REAL,
            height_cm REAL,
            age INTEGER,
            gender VARCHAR(50),
            activity_level VARCHAR(100),
            health_goals TEXT,
            dietary_restrictions TEXT,
            is_guest BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Detection history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_history (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT NOT NULL,
            detection_image_path TEXT,
            detected_foods JSONB,
            nutrition_data JSONB,
            health_score INTEGER,
            total_calories INTEGER,
            total_protein REAL,
            total_carbs REAL,
            total_fat REAL
        )
    ''')

    conn.commit()
    conn.close()

    print("✓ Database initialized (PostgreSQL)")


def hash_password(password):
    """Hash a password using bcrypt with salt."""
    if not password:
        return None
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')  # Store as string in database


def create_user(email=None, password=None, google_id=None, name=None, profile_picture=None, is_guest=False):
    """Create a new user account."""
    conn = get_connection()
    cursor = conn.cursor()

    password_hash = hash_password(password) if password else None

    try:
        cursor.execute('''
            INSERT INTO users (email, password_hash, google_id, name, profile_picture, is_guest)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        ''', (email, password_hash, google_id, name, profile_picture, is_guest))
        user_id = cursor.fetchone()[0]

        conn.commit()
        conn.close()
        return user_id
    except psycopg2.IntegrityError:
        conn.close()
        return None


def get_user_by_email(email):
    """Get user by email."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return dict(row)


def get_user_by_id(user_id):
    """Get user by ID."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return dict(row)


def get_user_by_google_id(google_id):
    """Get user by Google ID."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute('SELECT * FROM users WHERE google_id = %s', (google_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return dict(row)


def update_user_profile(user_id, **kwargs):
    """Update user profile information."""
    conn = get_connection()
    cursor = conn.cursor()

    # Build dynamic UPDATE query based on provided kwargs
    fields = []
    values = []

    allowed_fields = ['name', 'weight_kg', 'height_cm', 'age', 'gender',
                     'activity_level', 'health_goals', 'dietary_restrictions', 'profile_picture']

    for field in allowed_fields:
        if field in kwargs:
            fields.append(f"{field} = %s")
            values.append(kwargs[field])

    if not fields:
        conn.close()
        return False

    # Add updated_at timestamp
    fields.append("updated_at = %s")
    values.append(datetime.now().isoformat())
    values.append(user_id)

    query = f"UPDATE users SET {', '.join(fields)} WHERE id = %s"
    cursor.execute(query, values)

    conn.commit()
    conn.close()
    return True


def verify_password(email, password):
    """Verify user password using bcrypt."""
    user = get_user_by_email(email)
    if not user or not user.get('password_hash'):
        return None

    try:
        # Check if password matches the bcrypt hash
        if bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            return user
    except (ValueError, AttributeError):
        # Handle corrupted hash or encoding issues
        pass

    return None


def add_detection(image_path, detection_image_path, detected_foods, nutrition_data, user_id=None):
    """Add a new detection record to the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO detection_history
        (user_id, image_path, detection_image_path, detected_foods, nutrition_data,
         health_score, total_calories, total_protein, total_carbs, total_fat)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    ''', (
        user_id,
        image_path,
        detection_image_path,
        json.dumps(detected_foods),
        json.dumps(nutrition_data),
        nutrition_data.get('health_score', 0),
        nutrition_data.get('total_calories', 0),
        nutrition_data.get('total_protein', 0),
        nutrition_data.get('total_carbs', 0),
        nutrition_data.get('total_fat', 0)
    ))
    detection_id = cursor.fetchone()[0]

    conn.commit()
    conn.close()

    return detection_id


def get_recent_detections(limit=10, user_id=None):
    """Get recent detection history."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    if user_id is not None:
        cursor.execute('''
            SELECT id, timestamp, image_path, detection_image_path,
                   detected_foods, nutrition_data, health_score,
                   total_calories, total_protein, total_carbs, total_fat
            FROM detection_history
            WHERE user_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
        ''', (user_id, limit))
    else:
        cursor.execute('''
            SELECT id, timestamp, image_path, detection_image_path,
                   detected_foods, nutrition_data, health_score,
                   total_calories, total_protein, total_carbs, total_fat
            FROM detection_history
            ORDER BY timestamp DESC
            LIMIT %s
        ''', (limit,))

    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        row_dict = dict(row)

        # Parse JSON fields if they're strings (shouldn't be with JSONB but just in case)
        if isinstance(row_dict['detected_foods'], str):
            row_dict['detected_foods'] = json.loads(row_dict['detected_foods'])
        if isinstance(row_dict['nutrition_data'], str):
            row_dict['nutrition_data'] = json.loads(row_dict['nutrition_data'])

        history.append(row_dict)

    return history


def get_detection_by_id(detection_id):
    """Get a specific detection by ID."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute('SELECT * FROM detection_history WHERE id = %s', (detection_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    result = dict(row)

    # Parse JSON fields if they're strings
    if isinstance(result['detected_foods'], str):
        result['detected_foods'] = json.loads(result['detected_foods'])
    if isinstance(result['nutrition_data'], str):
        result['nutrition_data'] = json.loads(result['nutrition_data'])

    return result


def get_stats(user_id=None):
    """Get overall statistics."""
    conn = get_connection()
    cursor = conn.cursor()

    if user_id is not None:
        cursor.execute('''
            SELECT
                COUNT(*) as total_scans,
                COALESCE(SUM(jsonb_array_length(detected_foods)), 0) as total_foods,
                AVG(health_score) as avg_health_score,
                SUM(total_calories) as total_calories
            FROM detection_history
            WHERE user_id = %s
        ''', (user_id,))
    else:
        cursor.execute('''
            SELECT
                COUNT(*) as total_scans,
                COALESCE(SUM(jsonb_array_length(detected_foods)), 0) as total_foods,
                AVG(health_score) as avg_health_score,
                SUM(total_calories) as total_calories
            FROM detection_history
        ''')

    row = cursor.fetchone()
    conn.close()

    return {
        'total_scans': row[0] or 0,
        'total_foods_detected': row[1] or 0,
        'avg_health_score': round(row[2]) if row[2] else 0,
        'total_calories_tracked': row[3] or 0
    }


# Initialize database on import
init_database()
