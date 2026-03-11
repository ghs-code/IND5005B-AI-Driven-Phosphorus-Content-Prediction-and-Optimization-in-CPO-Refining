import pandas as pd
import statsmodels.api as sm
import os
import json
from itertools import combinations

# Define Paths
data_path1 = os.path.join("..", "data", "model_ready.csv")

# Load Dataset
df = pd.read_csv(data_path1)

dependent_var = 'feed_p_ppm'
independent_vars = ['feed_ffa_pct', 'feed_mi_pct', 'feed_iv', 'feed_dobi', 'feed_car_pv']

# Prepare Data
df_reg = df[[dependent_var] + independent_vars]

y = df_reg[dependent_var]

best_adj_r2 = -float('inf')
best_model = None
best_vars = []
run_log = []

# Iterate through all possible combinations of independent variables
for r in range(1, len(independent_vars) + 1):
    for combo in combinations(independent_vars, r):
        X_combo = df_reg[list(combo)]
        # Add constant to features
        X_combo = sm.add_constant(X_combo)
        
        # Fit OLS Model
        model = sm.OLS(y, X_combo).fit()
        
        # Check if it has the best adjusted R-squared
        if model.rsquared_adj > best_adj_r2:
            best_adj_r2 = model.rsquared_adj
            best_model = model
            best_vars = list(combo)
            
        run_log.append({
            "variables": list(combo),
            "rsquared": model.rsquared,
            "rsquared_adj": model.rsquared_adj,
            "aic": model.aic,
            "bic": model.bic
        })

# Output Results
out_path = "OLSresult_Optimal.csv"
with open(out_path, "w") as f:
    f.write(f"Best Independent Variables Selected: {', '.join(best_vars)}\n")
    f.write(f"Highest Adjusted R-squared: {best_adj_r2:.4f}\n\n")
    f.write(best_model.summary().as_csv())

print(f"Optimal OLS model selected with vars: {best_vars}")
print(f"Results successfully saved to {os.path.abspath(out_path)}")

log_path = "OLSresult_Optimal_Log.json"
with open(log_path, "w") as f:
    json.dump(run_log, f, indent=4)
print(f"Running log successfully saved to {os.path.abspath(log_path)}")
