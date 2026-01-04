import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("models/solar_model.pkl")

def main():
    st.set_page_config(page_title="Solar-Battery Hybrid Predictor", layout="wide")
    
    st.title("☀️🔋 Solar-Battery Hybrid Energy Prediction")
    st.markdown("""
    Predict the **power output** of your hybrid energy system using Machine Learning.
    Adjust the environmental and battery conditions below.
    """)
    
    # Sidebar for Inputs
    st.sidebar.header("Environment Conditions")
    
    irr = st.sidebar.slider("Solar Irradiation (kW/m²)", 0.0, 1.5, 0.8, 0.05)
    amb_temp = st.sidebar.slider("Ambient Temperature (°C)", 10.0, 50.0, 25.0, 0.5)
    mod_temp = st.sidebar.slider("Module Temperature (°C)", 10.0, 80.0, 35.0, 0.5)
    
    st.sidebar.header("Battery Status")
    soc = st.sidebar.slider("State of Charge (SOC %)", 0, 100, 80)
    voltage_factor = st.sidebar.slider("System Voltage (V) [Nominal 48V]", 40.0, 60.0, 48.0)
    
    # --- NEW: User Load & System Input ---
    st.sidebar.header("Home Energy Setup")
    system_capacity_kw = st.sidebar.slider("Solar System Size (kWp)", 1.0, 10.0, 5.0, 0.5)
    current_load = st.sidebar.slider("Current Load (kW)", 0.0, 5.0, 1.2, 0.1)
    
    # Inference
    try:
        model = load_model()
        input_data = pd.DataFrame({
            'AMBIENT_TEMPERATURE': [amb_temp],
            'MODULE_TEMPERATURE': [mod_temp],
            'IRRADIATION': [irr]
        })
        
        # Raw Prediction (Watts? Sum of Plant 1)
        # We validated max is ~29,000 (likely Watts -> 29 kW)
        # We need to scale this to the User's System Size
        raw_pred = model.predict(input_data)[0]
        
        # Normalize: (Prediction / Max_Plant_Capacity) * User_Capacity
        # Max observed ~ 30000 Watts
        PLANT_MAX_WATTS = 30000.0   
        efficiency = np.clip(raw_pred / PLANT_MAX_WATTS, 0.0, 1.0)
        
        predicted_solar_kw = efficiency * system_capacity_kw

    except FileNotFoundError:
        st.error("Model not found! Run 'python src/train_model.py' first.")
        predicted_solar_kw = 0
        raw_pred = 0
    
    # --- Energy Logic ---
    # Battery Specs (Estimated)
    # Voltage * Ah = Wh. / 1000 = kWh.
    battery_total_capacity_kwh = (voltage_factor * 100) / 1000 # ~4.8 kWh
    current_stored_energy_kwh = battery_total_capacity_kwh * (soc / 100.0)
    
    # Power Balance
    net_power = predicted_solar_kw - current_load
    
    # Helper for Time Formatting
    def format_time(hours):
        if hours > 24:
            return f"> 24 hours"
        elif hours >= 1.0:
            return f"{hours:.1f} hours"
        else:
            minutes = hours * 60
            return f"{int(minutes)} minutes"

    # Visual Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Solar Production", f"{predicted_solar_kw:.2f} kW", help=f"Scaled from {raw_pred:.0f}W Plant Data")
        
    with col2:
        st.metric("Home Consumption", f"{current_load:.2f} kW")
        
    with col3:
        # Net Status
        if net_power > 0.01:
            st.metric("Net Grid Export / Charge", f"+{net_power:.2f} kW", delta="Charging")
        elif net_power < -0.01:
            st.metric("Net Grid Import / Drain", f"{net_power:.2f} kW", delta="-Draining", delta_color="inverse")
        else:
            st.metric("Net Power", "0.00 kW", delta="Balanced", delta_color="off")

    st.divider()
    
    # --- Time Calculations (The Core Request) ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("🔋 Battery Status")
        st.progress(soc / 100.0, text=f"{soc}% Charged ({current_stored_energy_kwh:.2f} / {battery_total_capacity_kwh:.2f} kWh)")
        
        if net_power < -0.01:
            # Draining
            drain_rate_kw = abs(net_power)
            hours_left = current_stored_energy_kwh / drain_rate_kw
            time_str = format_time(hours_left)
            
            st.error(f"⚠️ Battery Draining! Time to Empty: **{time_str}**")
            st.caption(f"Based on current net drain of {drain_rate_kw:.2f} kW")
                 
        elif net_power > 0.01:
            # Charging
            charge_rate_kw = net_power
            energy_needed = battery_total_capacity_kwh - current_stored_energy_kwh
            
            # If full
            if energy_needed < 0.01:
                 st.success("Battery is Fully Charged!")
            else:
                hours_to_full = energy_needed / charge_rate_kw
                time_str = format_time(hours_to_full)
                st.success(f"⚡ Battery Charging! Time to Full: **{time_str}**")
                st.caption(f"Based on current net charge of {charge_rate_kw:.2f} kW")
                
        else:
            st.info("System is Balanced. Battery state holding.")

    # Visualizations
    st.divider()
    st.subheader("System Performance Curves")
    
    # Plot 1: Solar Curve vs Irradiation
    x_irr = np.linspace(0, 1.2, 50)
    # Fix temps to current slider
    dummy_df = pd.DataFrame({
        'AMBIENT_TEMPERATURE': [amb_temp]*50,
        'MODULE_TEMPERATURE': [mod_temp]*50,
        'IRRADIATION': x_irr
    })
    if 'model' in locals():
        y_pred = model.predict(dummy_df)
        chart_data = pd.DataFrame({'Irradiation': x_irr, 'Solar Output (kW)': y_pred})
        st.line_chart(chart_data.set_index('Irradiation'))
        st.caption("How Solar Output changes with Irradiation (at current temperatures)")

if __name__ == "__main__":
    main()
