import streamlit as st
import pandas as pd
import pickle
import folium
from streamlit_folium import st_folium

# --- 1. Page Configuration & Narrative ---
st.set_page_config(page_title="Thermal Twins | Urban Heat", page_icon="🌍", layout="wide")

st.title("🌍 Thermal Twins: Urban Heat Mitigation Engine")
st.markdown("""
Welcome to the predictive control center. This geospatial AI system allows policymakers 
to identify severe urban heat hotspots in Kolkata and dynamically simulate the cooling impact 
of physics-informed interventions like urban afforestation and reflective roofing.
""")
st.divider()

# --- 2. Load the Data Engine ---
@st.cache_data
def load_data():
    return pd.read_csv('kolkata_climate_data.csv')

@st.cache_resource
def load_model():
    with open('thermal_model.pkl', 'rb') as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# --- 3. The Interactive Sidebar (User Controls) ---
st.sidebar.header("⚙️ Scenario Simulation")
st.sidebar.markdown("Adjust the sliders below to apply environmental interventions and observe the thermodynamic response.")

# Sliders to simulate adding trees (NDVI) or reducing concrete heat absorption (NDBI)
tree_boost = st.sidebar.slider("🌳 Increase Green Cover (NDVI)", min_value=0.0, max_value=0.4, value=0.0, step=0.05)
roof_reduction = st.sidebar.slider("🏢 Apply Cool Roofs (Reduce NDBI)", min_value=0.0, max_value=0.4, value=0.0, step=0.05)

# --- 4. The AI Inference Engine ---
# Create a copy of the city data to apply the user's simulation
simulated_df = df.copy()
simulated_df['NDVI'] = simulated_df['NDVI'] + tree_boost
simulated_df['NDBI'] = simulated_df['NDBI'] - roof_reduction

# Ask the AI model to predict temperatures for BOTH the original city and the simulated city
original_preds = model.predict(df[['NDVI', 'NDBI']])
new_preds = model.predict(simulated_df[['NDVI', 'NDBI']])

# Calculate the actual drop in temperature across the city
temp_drop = original_preds.mean() - new_preds.mean()

# --- 5. Dashboard Metrics Presentation ---
col1, col2, col3 = st.columns(3)
col1.metric(label="Current Avg Surface Temp", value=f"{original_preds.mean():.2f} °C")
col2.metric(label="Simulated Avg Surface Temp", value=f"{new_preds.mean():.2f} °C", delta=f"-{temp_drop:.2f} °C", delta_color="inverse")
col3.metric(label="Total Datapoints Analyzed", value=f"{len(df):,}")

st.divider()

# --- 6. Geospatial Mapping ---
st.subheader("📍 High-Risk Heat Map Analysis (Kolkata)")
st.markdown("Red points indicate severe heat stress zones (> 37°C). Green points indicate natural cooling zones.")

# Initialize the map centered on Kolkata
m = folium.Map(location=[22.5726, 88.3639], zoom_start=11, tiles="CartoDB dark_matter")

# Sample 300 points to keep the web browser running smoothly
sample_df = df.sample(300, random_state=42)

for idx, row in sample_df.iterrows():
    # Determine dot color based on the current temperature prediction
    current_temp = model.predict([[row['NDVI'], row['NDBI']]])[0]
    
    if current_temp > 37:
        color = "#ff4b4b" # Red for severe heat
    elif current_temp > 34:
        color = "#ffa500" # Orange for moderate heat
    else:
        color = "#00cc66" # Green for cool zones

    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=f"Est. Temp: {current_temp:.1f}°C"
    ).add_to(m)

# Render the map in Streamlit
st_folium(m, width=1000, height=500, returned_objects=[])