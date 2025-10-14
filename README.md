Glucometer Glucose Forecaster 🩸
A web application built with Flask that uses a deep learning Transformer model to predict future blood glucose levels based on historical patient data from XML files.

Features
30-Minute Glucose Forecasting: Predicts a patient's blood glucose levels for the next 30 minutes.

XML File Upload: Accepts patient data in the specified XML format.

Interactive Visualization: Displays the patient's recent glucose history alongside the model's forecast on an interactive line chart using Chart.js.

Transformer Model: Utilizes a sophisticated Transformer-based architecture built with TensorFlow/Keras for time-series prediction.

Industry-Standard Project Structure: Organized using a scalable Flask Application Factory pattern with Blueprints.

Tech Stack
Backend: Python, Flask

Machine Learning: TensorFlow, Keras, Scikit-learn

Data Handling: Pandas, NumPy

Frontend: HTML, JavaScript, Chart.js

Project Structure
The project follows a professional Flask application factory structure for scalability and maintainability.

/glucometer-flask-app/
├── run.py                  # Main entry point to run the app
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
└── glucometer_app/         # The main application package
    ├── __init__.py         # Application factory
    ├── ml_utils.py         # Model loading and prediction logic
    ├── static/             # CSS/JS files
    ├── templates/          # HTML templates
    └── main/               # Main application blueprint
        ├── __init__.py
        └── routes.py

Setup and Installation
Follow these steps to get the application running on your local machine.

1. Clone the Repository
git clone <your-repository-url>
cd glucometer-flask-app

2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

Create the environment:

python -m venv venv

Activate the environment:

On Windows:

venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate

3. Install Dependencies
Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

4. Run the Application
Start the Flask development server by running the run.py file.

python run.py

The application will be available at http://127.0.0.1:5000.

How to Use
Open your web browser and navigate to http://127.0.0.1:5000.

Click the "Choose File" button and select a patient data XML file (a sample test_patient.xml is provided in the repository).

Click the "Get Prediction" button.

The application will process the data and display a chart showing the last 2 hours of historical glucose levels and the 30-minute forecast.

Model Information
The prediction model is a Transformer Encoder network built with TensorFlow and Keras. The architecture is defined in glucometer_app/ml_utils.py. The pre-trained model weights and architecture are saved in the glucometer_app/model/glucometer_transformer_model.h5 file.

License
This project is licensed under the MIT License. See the LICENSE file for details.
