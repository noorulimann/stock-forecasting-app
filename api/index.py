"""
Vercel Serverless Function Entry Point
Handles FastAPI app for Vercel deployment
"""

from app import app

# This is the handler that Vercel will call
# For FastAPI, we just export the app instance
handler = app
