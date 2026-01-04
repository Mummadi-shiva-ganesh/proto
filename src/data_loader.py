import pandas as pd
import numpy as np

def load_and_process_data(data_dir="data"):
    """
    Loads Solar and Battery data, merges them into a single dataframe
    aligned by sequential indices (simulating a hybrid system).
    """
    
    # 1. Load Solar Data
    # Files: solar_generation.csv, solar_weather.csv
    try:
        gen_df = pd.read_csv(f"{data_dir}/solar_generation.csv")
        weather_df = pd.read_csv(f"{data_dir}/solar_weather.csv")
    except FileNotFoundError:
        # Fallback to older names if renamed failed or for safety
        try:
             gen_df = pd.read_csv(f"{data_dir}/Plant_1_Generation_Data.csv")
             weather_df = pd.read_csv(f"{data_dir}/Plant_1_Weather_Sensor_Data.csv")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing solar data files in {data_dir}. Ensure 'solar_generation.csv' and 'solar_weather.csv' exist.") from e

    # Parse Dates
    # Format in Kaggle dataset often: '15-05-2020 00:00' or '2020-05-15 00:00:00'
    # strict=False allows mixed formats parsing
    gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], dayfirst=True)
    weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], dayfirst=True)

    # Merge Solar Generation and Weather
    gen_agg = gen_df.groupby('DATE_TIME').agg({
        'AC_POWER': 'sum',
        'DC_POWER': 'sum'
    }).reset_index()
    
    weather_agg = weather_df.groupby('DATE_TIME').agg({
        'AMBIENT_TEMPERATURE': 'mean',
        'MODULE_TEMPERATURE': 'mean',
        'IRRADIATION': 'mean'
    }).reset_index()
    
    solar_df = pd.merge(gen_agg, weather_agg, on='DATE_TIME', how='inner')
    
    # 2. Load Battery Data
    # File: battery_data.csv (Originally Experimental_data_fresh_cell.csv)
    # Headers expected: Time, Current, Voltage, ...
    batt_df = None
    try:
        batt_df = pd.read_csv(f"{data_dir}/battery_data.csv")
        
        # Check if legacy mock columns exist, else map new ones
        if 'SOC' not in batt_df.columns:
            # We have real experimental data (likely discharge)
            # Columns: Time, Voltage, Current, Temperature, ...
            # Clean headers (strip spaces)
            batt_df.columns = [c.strip() for c in batt_df.columns]
            
            # Simple SOC Estimation (Normalization of Time/Discharge)
            # Assuming the file is a full discharge cycle (fully charged start)
            # SOC = 100 * (1 - (Cumulative_Ah / Total_Capacity_Ah))
            # But simpler: Map Voltage to SOC? Or just normalize specific time range.
            # Let's simple normalize: Linearly map the first row to 100% and last row to 0%
            steps = len(batt_df)
            batt_df['SOC'] = np.linspace(100, 0, steps)
            
            # Map other columns for consistency
            column_map = {
                'Voltage': 'terminal_voltage',
                'Current': 'battery_current',
                'Temperature': 'battery_temp'
            }
            batt_df = batt_df.rename(columns=column_map)
            batt_df['ambient_temp'] = 25.0 # Default
            
    except FileNotFoundError:
        print("Warning: 'battery_data.csv' not found. Using Solar data only.")
        batt_df = pd.DataFrame() # Empty fallback
    
    # We'll trim to the length of the smaller dataset to ensure valid pairs
    min_len = min(len(solar_df), len(batt_df))
    
    solar_df = solar_df.iloc[:min_len].reset_index(drop=True)
    batt_df = batt_df.iloc[:min_len].reset_index(drop=True)
    
    # Concatenate columns
    combined_df = pd.concat([solar_df, batt_df], axis=1)
    
    # 3. Feature Engineering for "Hybrid" Prediction
    # Target: Total Energy Output potential
    # If Grid Load > Solar, Battery discharges.
    # We want to predict effective Energy Deliverable = Solar + (Battery Discharging Capacity)
    # But for a simple regression, let's predict: Solar AC Output (as a function of weather) 
    # OR predicted Battery SOC next step.
    
    # Let's predict AC_POWER based on available inputs, AND Battery Voltage/SOC?
    # Actually, the user wants "Hybrid Energy Prediction".
    # Let's define Target = AC_POWER (from solar) + (Battery_Voltage * Battery_Current(discharge) if needed).
    # For now, let's keep it simple: Predict 'AC_POWER' (Solar) separately?
    # Or predict 'System_Total_Power'?
    
    # Let's make the Model predict AC_POWER based on Irradiation/Temp (Verification of Solar)
    # AND predict Battery SOC based on use.
    
    # For a Regression task:
    # Predict AC_POWER using Irradiation, Temp
    # AND 
    # Predict Battery Voltage/SOC?
    
    # Let's target AC_POWER first as main energy source.
    
    return combined_df

if __name__ == "__main__":
    df = load_and_process_data()
    print("Data Loaded Successfully!")
    print(df.head())
    print(df.columns)
