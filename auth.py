from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user
import pickle
import os
from models import User, db
from flask_login import UserMixin

auth_bp = Blueprint('auth', __name__)

USERS_FILE = os.path.join(os.path.dirname(__file__), 'users.pkl')


class SimpleUser(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

# === Load and Save User Helpers ===
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'wb') as f:
        pickle.dump(users, f)

# === LOGIN ROUTE ===
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        users = load_users()

        user_data = users.get(username)
        if user_data and user_data['password'] == password:
            user = SimpleUser(id=user_data['id'], username=username)
            login_user(user, remember=True)
            flash("Login successful", "success")
            return redirect(url_for('upload'))  # Adjust route name if needed
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

# === REGISTER ROUTE ===
@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        users = load_users()

        if username in users:
            flash('Username already exists.', 'danger')
        else:
            user_id = len(users) + 1
            users[username] = {
                'id': user_id,
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'password': password
            }
            save_users(users)
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('auth.login'))

    return render_template('register.html')
