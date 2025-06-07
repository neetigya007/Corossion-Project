import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns

# Use your reference code to load and preprocess the data
electrochemical = pd.read_csv(r'C:\Users\neeti\OneDrive\Desktop\Electrochemical_Parameters.csv', encoding='utf-8')

# Replace problematic characters (e.g., Unicode minus sign)
electrochemical = electrochemical.applymap(lambda x: str(x).replace("\u2212", "-") if isinstance(x, str) else x)

# Rename columns: Replace Greek 'β' with 'Beta'
electrochemical.columns = [col.replace('β', 'Beta').strip() for col in electrochemical.columns]

electrochemical.rename(columns={
    "Temp (K)": "Temp",
    "Flowrate (L/min)": "Flowrate",
    "Ecorr (mV) (500ppm)": "EcorrNew",
    "Ecorr (mV) (Blank)": "Ecorr",
    "−Betac (V/dec)": "Betac (V/dec)"
}, inplace=True)

# Convert all numeric columns to proper numeric datatypes
numeric_cols = ["Temp", "Flowrate", "Ecorr", "EcorrNew", "Betaa (V/dec)", "Betac (V/dec)"]
for col in numeric_cols:
    electrochemical[col] = pd.to_numeric(electrochemical[col], errors='coerce')

# Display the original dataset
print("Original Dataset:")
print(electrochemical)
print("\nDataset Shape:", electrochemical.shape)

# ========== INTERPOLATION-BASED VSG ==========

# Define the number of temperature points to generate
num_temp_points = 21  # This will give us a 1K resolution between 303K and 323K

# Temperature range constraints
temp_min, temp_max = 303, 323
new_temps = np.linspace(temp_min, temp_max, num_temp_points)

# Initialize list to store expanded data
expanded_data = []

# For each unique flowrate in the original dataset
for flowrate in electrochemical['Flowrate'].unique():
    # Filter data for this flowrate
    flow_data = electrochemical[electrochemical['Flowrate'] == flowrate]
    
    # Sort by temperature for proper interpolation
    flow_data = flow_data.sort_values('Temp')
    
    # Create interpolation functions for each column (except Flowrate)
    interp_funcs = {}
    for col in electrochemical.columns:
        if col != 'Flowrate':
            # Use linear interpolation for all columns
            interp_funcs[col] = interp1d(
                flow_data['Temp'], 
                flow_data[col], 
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
    
    # Generate interpolated samples for each temperature point
    for temp in new_temps:
        # Check if this is an original point
        is_original = any(abs(temp - orig_temp) < 1e-6 for orig_temp in flow_data['Temp'])
        
        if is_original:
            # Use the original data point
            orig_point = flow_data[abs(flow_data['Temp'] - temp) < 1e-6].iloc[0].to_dict()
            orig_point['Data_Type'] = 'Original'
            expanded_data.append(orig_point)
        else:
            # Create interpolated data point
            new_point = {'Flowrate': flowrate, 'Data_Type': 'Interpolated'}
            for col in interp_funcs:
                new_point[col] = float(interp_funcs[col](temp))
            expanded_data.append(new_point)

# Convert to DataFrame
expanded_df = pd.DataFrame(expanded_data)

# Display expanded dataset stats
print("\nExpanded Dataset:")
print(f"Original size: {len(electrochemical)} samples")
print(f"Expanded size: {len(expanded_df)} samples")
print(f"Expansion factor: {len(expanded_df)/len(electrochemical):.2f}x")
print("\nSample of expanded dataset:")
print(expanded_df.head(10))

# ========== VISUALIZATION ==========

# Use a more professional color scheme
colors = sns.color_palette("viridis", 3)
flowrate_colors = {4: colors[0], 8: colors[1], 12: colors[2]}

# Create visualizations
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Ecorr (Blank) vs Temperature
for flowrate in expanded_df['Flowrate'].unique():
    # Filter by flowrate
    flow_data = expanded_df[expanded_df['Flowrate'] == flowrate]
    
    # Get original points
    orig = flow_data[flow_data['Data_Type'] == 'Original']
    # Get interpolated points
    interp = flow_data[flow_data['Data_Type'] == 'Interpolated']
    
    # Sort for line plot
    flow_data_sorted = flow_data.sort_values('Temp')
    
    # Plot line connecting all points
    axes[0].plot(flow_data_sorted['Temp'], flow_data_sorted['Ecorr'], 
                color=flowrate_colors[flowrate], alpha=0.7,
                label=f'Flow={flowrate} L/min')
    
    # Plot original points with larger markers
    axes[0].scatter(orig['Temp'], orig['Ecorr'], 
                   color=flowrate_colors[flowrate], s=100, edgecolor='black')
    
    # Plot interpolated points with smaller markers
    axes[0].scatter(interp['Temp'], interp['Ecorr'], 
                   color=flowrate_colors[flowrate], alpha=0.5, s=30)

axes[0].set_title('Temperature vs Ecorr (Blank)', fontsize=14)
axes[0].set_xlabel('Temperature (K)', fontsize=12)
axes[0].set_ylabel('Ecorr (mV) (Blank)', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Ecorr (500ppm) vs Temperature
for flowrate in expanded_df['Flowrate'].unique():
    # Filter by flowrate
    flow_data = expanded_df[expanded_df['Flowrate'] == flowrate]
    
    # Get original points
    orig = flow_data[flow_data['Data_Type'] == 'Original']
    # Get interpolated points
    interp = flow_data[flow_data['Data_Type'] == 'Interpolated']
    
    # Sort for line plot
    flow_data_sorted = flow_data.sort_values('Temp')
    
    # Plot line connecting all points
    axes[1].plot(flow_data_sorted['Temp'], flow_data_sorted['EcorrNew'], 
                color=flowrate_colors[flowrate], alpha=0.7,
                label=f'Flow={flowrate} L/min')
    
    # Plot original points with larger markers
    axes[1].scatter(orig['Temp'], orig['EcorrNew'], 
                   color=flowrate_colors[flowrate], s=100, edgecolor='black')
    
    # Plot interpolated points with smaller markers
    axes[1].scatter(interp['Temp'], interp['EcorrNew'], 
                   color=flowrate_colors[flowrate], alpha=0.5, s=30)

axes[1].set_title('Temperature vs Ecorr (500ppm)', fontsize=14)
axes[1].set_xlabel('Temperature (K)', fontsize=12)
axes[1].set_ylabel('Ecorr (mV) (500ppm)', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('Expanded_Dataset_Visualization.png', dpi=300)
plt.show()

# Save expanded dataset to CSV
expanded_df.to_csv('Expanded_Electrochemical_Parameters.csv', index=False)
print("\nExpanded dataset saved to 'Expanded_Electrochemical_Parameters.csv'")
