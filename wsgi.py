"""
wsgi.py — Production WSGI entry point for SEAD-AI
Used by Gunicorn, Render, Heroku, etc.
Usage: gunicorn wsgi:app
"""
from app import app

if __name__ == "__main__":
    app.run()
