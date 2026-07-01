# 🌍 Thermal Trinity: Urban Heat Mitigation Engine

## 📌 The Problem
Urban Heat Islands (UHI) severely impact public health and energy consumption in metropolitan areas like Kolkata. Traditional monitoring identifies heat but fails to offer predictive, actionable mitigation strategies.

## 🚀 Our Solution
Thermal Trinity is a spatial AI platform designed to simulate the thermodynamic impact of environmental interventions. By generating a synthetic, physics-informed climate matrix encompassing NDVI (vegetation) and NDBI (built-up density), our machine learning model predicts Land Surface Temperature (LST).

## ⚙️ Technical Architecture
* **Data Engine:** Python, Pandas, Numpy (Physics-informed synthetic data generation)
* **Machine Learning:** Scikit-Learn (Random Forest Spatial Regressor)
* **Frontend Dashboard:** Streamlit, Folium (Interactive mapping and real-time simulation)

## 💻 How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the data engine: `python engine.py`
4. Launch the dashboard: `streamlit run app.py`
