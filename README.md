# ☀️🔋 Solar-Battery Hybrid Energy Predictor

A generic ML-powered web application to predict solar power generation and battery usage for hybrid energy systems.

## 📌 Project Overview
This tool helps users predict:
-   **Solar Power Output** based on environmental conditions (Irradiance, Temperature).
-   **Battery Status**: Charging time or time-to-empty based on home load.
-   **Net Power**: Visualizes real-time Grid Export (Charging) or Import (Draining).

The application uses a **Client-Server Architecture**:
-   **Backend**: Python (Flask) API serving a Machine Learning model (Random Forest).
-   **Frontend**: Modern HTML5 Dashboard styled with Tailwind CSS.

---

## 🚀 Installation Guide

### Prerequisites
-   Python 3.8 or higher installed.
-   Git (to clone the repository).

### 1. Clone the Repository
```bash
git clone https://github.com/Mummadi-shiva-ganesh/proto.git
cd proto
```

### 2. Install Dependencies
Install the required Python libraries using pip:
```bash
pip install -r requirements.txt
```
*Note: This installs Flask, scikit-learn, pandas, flask-cors, etc.*

---

## ⚡ How to Run the App

You need to run the Python Backend first, then open the Frontend.

### Step 1: Start the Backend API
Open your terminal/command prompt in the project folder and run:
```bash
python src/api.py
```
> **Keep this terminal open!** You should see a message saying `Running on http://0.0.0.0:5000`. This functionality powers the AI predictions.

### Step 2: Open the Dashboard
1.  Locate the `index.html` file in the main project folder.
2.  **Double-click** `index.html` to open it in your browser (Chrome, Edge, etc.).

---

## 🎮 How to Use
1.  **Environment Controls**: Adjust sliders for *Irradiation* and *Temperature* to simulate different weather conditions.
2.  **System Config**: Set your *Solar System Size* (e.g., 5kW) and *Current Home Load* (e.g., 1.2kW).
3.  **Real-Time Output**:
    -   **Net Power Card**: Turns **Green** if you are exporting/charging, **Red** if draining.
    -   **Battery Status**: Shows estimated **Time to Full** or **Time to Empty**.

## 📁 Project Structure
-   `src/api.py` - Flask Server & API Logic.
-   `src/train_model.py` - Script used to train the ML model.
-   `models/` - Contains the trained `.pkl` model file.
-   `data/` - Dataset files (Solar & Battery).
-   `index.html` - Main Dashboard UI.
-   `script.js` - Frontend logic (talks to the API).

---

## 🛠️ Troubleshooting
-   **"Error connecting to server"**: Make sure `src/api.py` is running in a terminal window.
-   **Wrong Predictions?**: Ensure the `models/solar_model.pkl` file exists. If not, run `python src/train_model.py` to regenerate it.
