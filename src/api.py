from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app) # Enable CORS for all domains

# Load Model
MODEL_PATH = "models/solar_model.pkl"
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract Inputs
        irr = float(data.get('irradiation', 0.8))
        amb_temp = float(data.get('ambient_temp', 25.0))
        mod_temp = float(data.get('module_temp', 35.0))
        
        system_capacity_kw = float(data.get('system_capacity', 5.0))
        current_load = float(data.get('load', 1.2))
        
        battery_soc_percent = float(data.get('soc', 80.0))
        voltage_factor = float(data.get('voltage', 48.0))
        
        if model is None:
            return jsonify({'error': 'Model not loaded. Train model first.'}), 500
        
        # 1. Solar Prediction
        input_df = pd.DataFrame({
            'AMBIENT_TEMPERATURE': [amb_temp],
            'MODULE_TEMPERATURE': [mod_temp],
            'IRRADIATION': [irr]
        })
        
        # Raw Prediction (Watts? Sum of Plant 1)
        # Validated max is ~30kW (30000 Watts) from original dataset
        raw_pred_watts = model.predict(input_df)[0]
        
        # Normalize to User System Size
        PLANT_MAX_WATTS = 30000.0   
        efficiency = np.clip(raw_pred_watts / PLANT_MAX_WATTS, 0.0, 1.0)
        
        solar_kw = efficiency * system_capacity_kw
        
        # 2. Battery Logic
        battery_total_capacity_kwh = (voltage_factor * 100) / 1000 # ~4.8 kWh nominal (Assuming 100Ah)
        current_stored_energy_kwh = battery_total_capacity_kwh * (battery_soc_percent / 100.0)
        
        # 3. Net Power & Time Stats
        net_power = solar_kw - current_load
        
        time_msg = "Balanced"
        status = "balanced" # charging, draining, balanced
        
        if net_power < -0.01:
            # Draining
            status = "draining"
            drain_rate_kw = abs(net_power)
            hours_left = current_stored_energy_kwh / drain_rate_kw if drain_rate_kw > 0 else 0
            
            # Format time
            minutes_left = int(hours_left * 60)
            if hours_left >= 1.0:
                time_msg = f"{hours_left:.1f} hours to Empty"
            else:
                time_msg = f"{minutes_left} minutes to Empty"
                
        elif net_power > 0.01:
            # Charging
            status = "charging"
            charge_rate_kw = net_power
            energy_needed = battery_total_capacity_kwh - current_stored_energy_kwh
            
            if energy_needed < 0.01:
                time_msg = "Fully Charged"
                status = "full"
            else:
                hours_to_full = energy_needed / charge_rate_kw
                minutes_to_full = int(hours_to_full * 60)
                
                if hours_to_full >= 1.0:
                    time_msg = f"{hours_to_full:.1f} hours to Full"
                else:
                    time_msg = f"{minutes_to_full} minutes to Full"
        
        return jsonify({
            'solar_kw': round(solar_kw, 2),
            'net_power': round(net_power, 2),
            'battery_kwh': round(current_stored_energy_kwh, 2),
            'battery_total_kwh': round(battery_total_capacity_kwh, 2),
            'status': status,
            'time_msg': time_msg
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Flask API on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
