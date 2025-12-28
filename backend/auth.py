"""
Authentication utilities for user management
"""
import os
import requests
from functools import wraps
from flask import session, redirect, url_for, jsonify


def login_required(f):
    """Decorator to require login for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function


def get_current_user_id():
    """Get the current logged-in user ID from session."""
    return session.get('user_id')


def verify_google_token(token):
    """
    Verify Google OAuth token and return user info.

    Args:
        token: Google ID token

    Returns:
        dict with user info or None if invalid
    """
    try:
        # Verify token with Google
        response = requests.get(
            'https://www.googleapis.com/oauth2/v3/tokeninfo',
            params={'id_token': token}
        )

        if response.status_code == 200:
            user_info = response.json()

            # Verify the token is for our app
            client_id = os.getenv('GOOGLE_CLIENT_ID')
            if client_id and user_info.get('aud') != client_id:
                return None

            return {
                'google_id': user_info.get('sub'),
                'email': user_info.get('email'),
                'name': user_info.get('name'),
                'picture': user_info.get('picture')
            }
        return None
    except Exception as e:
        print(f"Error verifying Google token: {e}")
        return None
