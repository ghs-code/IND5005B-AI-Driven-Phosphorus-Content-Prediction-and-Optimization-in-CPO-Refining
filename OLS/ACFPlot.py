import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import os

# Define paths relative to the script location
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data", "model_ready_lag.csv")
output_path = os.path.join(current_dir, "output", "acf_plot.jpg")

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load data
df = pd.read_csv(data_path)

# Extract the dependent variable and drop any missing values 
# plot_acf requires data without NaNs
feed_p_ppm = df['feed_p_ppm'].dropna()

# Create the plot with specified figure size
fig, ax = plt.subplots(figsize=(10, 6))

# Plot ACF with 40 lags, 95% confidence interval (alpha=0.05), and include lag=0
plot_acf(feed_p_ppm, lags=40, alpha=0.05, zero=True, ax=ax)

# Set titles and labels according to requirements
ax.set_title("Autocorrelation Function (ACF) of Daily Feed Phosphorus Content (ppm)")
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Autocorrelation")

# Improve layout
plt.tight_layout()

# Save plot to jpg format in the output folder
plt.savefig(output_path, format='jpg', dpi=300)

# Close the plot to free memory
plt.close()

print(f"ACF plot successfully saved to: {os.path.abspath(output_path)}")
