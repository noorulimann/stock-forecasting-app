from flask import Flask, render_template, request, jsonify
from flask_pymongo import PyMongo
from config import Config
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Fix Flask-PyMongo configuration
    app.config['MONGO_URI'] = app.config['MONGODB_URI']
    
    # Initialize MongoDB
    mongo = PyMongo(app)
    
    # Store mongo instance in app for access in routes
    app.mongo = mongo
    
    # Import routes
    from routes import main
    app.register_blueprint(main)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=app.config['DEBUG'], port=app.config['PORT'])