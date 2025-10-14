from flask import Flask
from config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Register your blueprints here
    from glucometer_app.main.routes import main
    app.register_blueprint(main)

    return app