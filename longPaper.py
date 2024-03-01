import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import linear_reset
from scipy.stats import f
from statsmodels.compat import lzip
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import anderson
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_harvey_collier





# Include similar code as HW1 for the basic steps
file_location = "/Users/amyliang/Eco441K/HW10/longPaper.xlsx"
df= pd.read_excel(file_location)

pd.set_option('display.max_columns', None)



print(df.describe())


# Extracting the dependent variable (average home value)
y = df['Average Home Value, 2019']

# Extracting independent variables (excluding 'West')
X = df[['Unemployed, 2023, % ', 'Education', 'Median annual income, 2021', 'Highest Interest, 2023,  %', 'South', 'Midwest', 'Northeast']]

# Adding a constant term to the independent variables
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression
print(model.summary())
print()

# Get predicted values and residuals
predicted_values = model.fittedvalues
residuals = model.resid

# Create a scatter plot of residuals against predicted values
plt.scatter(predicted_values, residuals)
plt.axhline(y=0, color='red', linestyle='--')  # Add a horizontal line at y=0
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Run the Breusch-Pagan test
bp_test = het_breuschpagan(model.resid, model.model.exog)

print()
print()

# Extract the p-value
p_value = bp_test[1]

# Test if it passed or failed the heteroskedasticity 
if p_value < 0.05:
    print("Failed. The Breusch-Pagan test is statistically significant, indicating the presence of heteroskedasticity.")
else:
    print("Passed. The Breusch-Pagan test is not statistically significant, suggesting no evidence of heteroskedasticity.")
print()

# Calculate VIF for each variable
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display the VIF values
print(vif_data)
print()

print()

# Specify the number of initial observations to skip
skip = 5  # You may adjust this value based on your data

# Perform the Ramsey RESET test with the specified skip value
reset_test = linear_harvey_collier(model, skip=skip)

# Print the test results
print("Ramsey RESET Test:")
print(reset_test)

# Extract the p-value
p_value = reset_test.pvalue

# Check if the p-value is less than the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The Ramsey RESET test rejects the null hypothesis.")
    print("There may be omitted variable bias or misspecification.")
else:
    print("The Ramsey RESET test does not reject the null hypothesis.")
    print("There is no strong evidence of omitted variable bias or misspecification.")