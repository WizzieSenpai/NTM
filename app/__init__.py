from flask import Flask
from pathlib import Path

def create_app(test_config=None):
    """
    Application factory function to create and configure the Flask app
    
    Args:
        test_config: Configuration to use for testing (optional)
    
    Returns:
        Flask application instance
    """
    # Create Flask app
    app = Flask(__name__, instance_relative_config=True)
    
    # Set default configuration
    app.config.from_mapping(
        SECRET_KEY='dev',
        MODEL_PATH=str(Path(__file__).parent.parent / 'models' / 'translation_model.h5'),
        DATA_PATH=str(Path(__file__).parent.parent / 'data')
    )
    
    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.update(test_config)
    
    # Ensure the instance folder exists
    try:
        Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating instance path: {e}")
    
    # Initialize extensions
    # Add any Flask extensions here (e.g., SQLAlchemy, Login Manager, etc.)
    
    return app

# Version information
__version__ = '1.0.0'