from flask import render_template, request, Blueprint
import pandas as pd
from glucometer_app.ml_uitls import process_xml_data, make_prediction

main = Blueprint('main', __name__)

@main.route("/")
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    if 'xmlfile' not in request.files:
        return "No file part", 400
    file = request.files['xmlfile']
    if not file or file.filename == '':
        return "No selected file", 400
    
    # Process the data using the utility function
    processed_df = process_xml_data(file)
    
    # Make prediction using the utility function
    prediction, history = make_prediction(processed_df)
    
    # Prepare data for rendering
    history_timestamps = history.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
    history_values = history.values.tolist()
    
    last_timestamp = history.index[-1]
    prediction_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=5), periods=6, freq='5T')
    prediction_timestamps_str = prediction_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist()
    
    return render_template('result.html', 
                           history_labels=history_timestamps,
                           history_data=history_values,
                           pred_labels=prediction_timestamps_str,
                           pred_data=prediction.tolist())