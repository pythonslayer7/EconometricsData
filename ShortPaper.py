import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Short paper code
# Include similar code as HW1 for the basic steps
file_location = "/Users/amyliang/Downloads/ShortPaperData.xlsx"
# Display all columns.
pd.set_option('display.max_columns', None)
# Read data
new_df= pd.read_excel(file_location)


# Replace "insufficient data" with NaN in the "Inactivity" column
new_df["Inactivity"].replace("Insufficient data**", float("nan"), inplace=True)

new_df = new_df.dropna()

print(new_df.describe())

# Specify the independent variables
independent_vars = ['Education', 'Median annual income, 2021', 'Uninsured, 2021,  %', 'South', 'Midwest', 'Northeast', 'Inactivity']

# Check if the specified independent variables exist in the DataFrame
missing_vars = set(independent_vars) - set(new_df.columns)
if missing_vars:
    # Print the missing variables and the actual column names in the DataFrame
    print(f"Error: The following variables are not present in the DataFrame: {missing_vars}")
    print(f"Actual column names in the DataFrame: {new_df.columns}")
else:
    # Create the design matrix (X) and the target variable (y)
    X = new_df[independent_vars]
    y = new_df['Obesity, %, 2021']

    # Add a constant to the design matrix (required for statsmodels)
    X = sm.add_constant(X)

    # Fit the linear regression model
    model = sm.OLS(y, X).fit()

    # Print the regression summary
    print(model.summary())