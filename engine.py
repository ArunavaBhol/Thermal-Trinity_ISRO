import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle

# Set random seed for perfectly reproducible results
np.random.seed(42)

def generate_local_kolkata_data(num_samples=2000):
    """
    Generates a highly realistic spatial climate matrix for Kolkata
    modeled directly on real urban heat island thermodynamic principles.
    """
    print("Step 1: Generating local geographic coordinates for Kolkata...")
    # Bound boxes tightly around the Kolkata metropolitan area
    latitudes = np.random.uniform(22.5000, 22.6200, num_samples)
    longitudes = np.random.uniform(88.3300, 88.4300, num_samples)
    
    print("Step 2: Simulating environmental indices (NDVI and NDBI)...")
    # NDVI (Vegetation Index): ranges from -0.1 (water/concrete) to 0.5 (dense trees)
    ndvi = np.random.uniform(-0.1, 0.5, num_samples)
    
    # NDBI (Built-up Index): high concrete density usually correlates with lower vegetation
    ndbi = -0.7 * ndvi + np.random.normal(0.2, 0.1, num_samples)
    ndbi = np.clip(ndbi, -0.3, 0.8) # Keep within realistic satellite boundaries

    print("Step 3: Computing Land Surface Temperature (LST) using thermodynamic rules...")
    # Base summer temperature for the region is around 36°C
    base_temp = 36.0
    
    # Physics rule: Vegetation cools the surface, concrete (built-up) heats it up intensely
    cooling_effect = -12.0 * ndvi
    heating_effect = 8.5 * ndbi
    environmental_noise = np.random.normal(0, 1.2, num_samples) # Simulates microclimate variations
    
    lst = base_temp + cooling_effect + heating_effect + environmental_noise
    
    # Structure into a clean analytics dataframe
    df = pd.DataFrame({
        'Longitude': longitudes,
        'Latitude': latitudes,
        'NDVI': ndvi,
        'NDBI': ndbi,
        'LST': lst
    })
    return df

def train_thermal_model(df):
    """
    Trains a Spatial AI Regressor to learn how environmental modifications
    impact local surface temperatures.
    """
    print("\nStep 4: Separating features and targeting vectors...")
    X = df[['NDVI', 'NDBI']]
    y = df['LST']
    
    # FIXED: Corrected parameter name from test_test_split to test_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Step 5: Training the Random Forest Spatial Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    predictions = model.predict(X_test)
    accuracy = r2_score(y_test, predictions)
    print(f"-> Model Training Complete. R-Squared Accuracy Score: {accuracy:.4f}")
    
    # Save the trained model locally as a reusable binary file
    with open('thermal_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("-> Model successfully serialized and saved to 'thermal_model.pkl'")
    
    return model

if __name__ == "__main__":
    print("=== STARTING LOCAL THERMAL TWINS DATA & ML PIPELINE ===")
    dataset = generate_local_kolkata_data()
    
    print("\n--- Preview of Generated Climate Matrix ---")
    print(dataset.head())
    
    # Save the raw dataset to a local CSV file for the dashboard to read later
    dataset.to_csv('kolkata_climate_data.csv', index=False)
    print("\nDataset successfully backed up to 'kolkata_climate_data.csv'")
    
    # Train and save the model
    train_thermal_model(dataset)
    print("\n=== PIPELINE RUN SUCCESSFUL: ZERO CLOUD OVERHEAD ===")