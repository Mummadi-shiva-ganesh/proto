import pandas as pd
import numpy as np
import os

def create_solar_data():
    print("Generating Mock Solar Data...")
    # Create time range
    dates = pd.date_range(start='2020-05-15', end='2020-06-17', freq='15T')
    n_samples = len(dates)
    
    # 1. Generation Data
    # Columns: DATE_TIME, PLANT_ID, SOURCE_KEY, DC_POWER, AC_POWER, DAILY_YIELD, TOTAL_YIELD
    plant_id = 4135001
    source_keys = [f'INV_{i}' for i in range(1, 4)] # Simulate 3 inverters
    
    gen_data_list = []
    
    for key in source_keys:
        # Simulate simple day-night cycle for power
        # Peak at noon
        hour_scaler = np.clip(np.sin((dates.hour - 6) * np.pi / 12), 0, 1)
        random_variation = np.random.normal(1, 0.1, n_samples)
        
        ac_power = 1000 * hour_scaler * random_variation
        ac_power = np.clip(ac_power, 0, None) # No negative power
        dc_power = ac_power * 10 # Roughly 10x DC via inverter
        
        daily_yield = np.cumsum(ac_power) # Simplified cumulative
        total_yield = np.cumsum(ac_power) + 100000 # Offset
        
        df_inv = pd.DataFrame({
            'DATE_TIME': dates,
            'PLANT_ID': plant_id,
            'SOURCE_KEY': key,
            'DC_POWER': dc_power,
            'AC_POWER': ac_power,
            'DAILY_YIELD': daily_yield,
            'TOTAL_YIELD': total_yield
        })
        gen_data_list.append(df_inv)
        
    gen_df = pd.concat(gen_data_list, ignore_index=True)
    
    # 2. Weather Data
    # Columns: DATE_TIME, PLANT_ID, SOURCE_KEY, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION
    # Note: Weather is usually one sensor per plant, but schema says SOURCE_KEY? Often it's just one.
    # Let's assume one sensor.
    
    # Temp peaks at 2pm usually
    temp_curve = np.clip(np.sin((dates.hour - 8) * np.pi / 12), 0, 1)
    ambient_temp = 25 + 10 * temp_curve + np.random.normal(0, 1, n_samples)
    module_temp = ambient_temp + (20 * temp_curve) # Modules get hotter
    
    # Irradiation peaks at noon
    irr_curve = np.clip(np.sin((dates.hour - 6) * np.pi / 12), 0, 1)
    irradiation = 1.0 * irr_curve * np.random.normal(1, 0.05, n_samples)
    irradiation = np.clip(irradiation, 0, None)
    
    weather_df = pd.DataFrame({
        'DATE_TIME': dates,
        'PLANT_ID': plant_id,
        'SOURCE_KEY': 'SENSOR_1',
        'AMBIENT_TEMPERATURE': ambient_temp,
        'MODULE_TEMPERATURE': module_temp,
        'IRRADIATION': irradiation
    })
    
    return gen_df, weather_df

def create_battery_data():
    print("Generating Mock Battery Data...")
    # Kaggle Battery Dataset structure
    # timestamp, SOC, SOH, terminal_voltage, battery_current, battery_temp, internal_resistance, ambient_temp, ...
    
    # Generate 5000 random samples (arbitrary time steps)
    n_samples = 5000
    timestamps = np.arange(n_samples)
    
    soc = np.linspace(100, 0, n_samples) # Linear discharge
    soc = soc + np.random.normal(0, 2, n_samples)
    soc = np.clip(soc, 0, 100)
    
    voltage = 3.2 + (soc / 100.0) * 1.0 # Simple linear voltage curve
    current = np.random.normal(-10, 2, n_samples) # Discharging mostly
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'SOC': soc,
        'terminal_voltage': voltage,
        'battery_current': current,
        'battery_temp': 30 + np.random.normal(0, 2, n_samples),
        'ambient_temp': 25 + np.random.normal(0, 1, n_samples),
        'SOH': 1.0 # Healthy
        # Add other cols if strictly needed, but these are main ones
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    gen_df, weather_df = create_solar_data()
    gen_df.to_csv("data/Plant_1_Generation_Data.csv", index=False)
    weather_df.to_csv("data/Plant_1_Weather_Sensor_Data.csv", index=False)
    
    batt_df = create_battery_data()
    batt_df.to_csv("data/battery_data.csv", index=False)
    
    print("Mock datasets created successfully in 'data/' folder.")
