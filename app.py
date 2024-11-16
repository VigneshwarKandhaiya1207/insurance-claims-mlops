from flask import Flask, render_template, request, redirect, url_for
from src.Insurance.pipeline.predication_pipeline import PredictionPipeline
from logger import logger
import joblib
import os
import numpy as np
import pandas as pd
app = Flask(__name__)

# Load your model once (assuming a trained model is saved as 'model.pkl')


# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        action = request.form['action']
        if action == 'train':
            return redirect(url_for('train_model'))
        elif action == 'predict':
            return redirect(url_for('prediction_form'))
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    action = request.form['action']
    if action == 'train':
        return redirect(url_for('train_model'))
    elif action == 'predict':
        return redirect(url_for('prediction_form'))
    return redirect(url_for('index'))  # Default to the index page if something goes wrong


# Training route
@app.route('/train', methods=['GET'])
def train_model():
    print("Training started...")
    os.system("python main.py")
    return "Training started successfully!" 

# Prediction route (GET to show the form, POST to process it)
@app.route('/predict', methods=['GET', 'POST'])
def prediction_form():
    if request.method == 'POST':

        data = {
            'policy_tenure': float(request.form["policy_tenure"]),
            'age_of_car': float(request.form["age_of_car"]),
            'age_of_policyholder': float(request.form["age_of_policyholder"]),
            'population_density': int(request.form["population_density"]),
            'displacement': int(request.form["displacement"]),
            'turning_radius': float(request.form["turning_radius"]),
            'length': int(request.form["length"]),
            'width': int(request.form["width"]),
            'height': int(request.form["height"]),
            'gross_weight': int(request.form["gross_weight"]),
            'area_cluster': request.form["area_cluster"],
            'segment': request.form["segment"],
            'model': request.form["model"],
            'fuel_type': request.form["fuel_type"],
            'max_torque': request.form["max_torque"],
            'max_power': request.form["max_power"],
            'steering_type': request.form["steering_type"],
            'is_esc': request.form["is_esc"],
            'is_adjustable_steering': request.form["is_adjustable_steering"],
            'is_parking_sensors': request.form["is_parking_sensors"],
            'is_front_fog_lights': request.form["is_front_fog_lights"],
            'is_rear_window_wiper': request.form["is_rear_window_wiper"],
            'is_rear_window_washer': request.form["is_rear_window_washer"],
            'is_rear_window_defogger': request.form["is_rear_window_defogger"],
            'is_brake_assist': request.form["is_brake_assist"],
            'is_power_door_locks': request.form["is_power_door_locks"],
            'is_central_locking': request.form["is_central_locking"],
            'is_driver_seat_height_adjustable': request.form["is_driver_seat_height_adjustable"],
            'is_day_night_rear_view_mirror': request.form["is_day_night_rear_view_mirror"],
            'is_ecw': request.form["is_ecw"],
            'cylinder': int(request.form["cylinder"]),
            'ncap_rating': int(request.form["ncap_rating"]),
            'airbags': int(request.form["airbags"])
        }

        df = pd.DataFrame([data])
        logger.info(df)
        logger.info(df.info())
        pred_model=PredictionPipeline()
        prediction = pred_model.predict(df)

        return f"Prediction Result: {prediction}"

    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
