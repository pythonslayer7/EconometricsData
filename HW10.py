import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.api as sm
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_breuschpagan


# Include similar code as HW1 for the basic steps
file_location = "/Users/amyliang/Eco441K/HW10/intdef.dta"
f2 = "/Users/amyliang/Eco441K/HW10/BARIUM.DTA"
f3 = "/Users/amyliang/Eco441K/HW10/loanapp.dta"
df= pd.read_stata(file_location)
new_df = pd.read_stata(f2)
df2 = pd.read_stata(f3)

pd.set_option('display.max_columns', None)

# Wooldridge Computer Exercise C7.8
print("Wooldridge Computer Exercise C7.8")

# Show all variable names
print(df2.describe())

# Define the independent variable (white) and the dependent variable (approve)
independent_var = 'white'
dependent_var = 'approve'

# Prepare the independent variable (X) and the dependent variable (y)
X = sm.add_constant(df2[independent_var])
y = df2[dependent_var]

# Fit the linear probability model
model = sm.OLS(y, X)
results = model.fit()

# Print regression results
print(results.summary())
print()

a1 = """
(i)Even without regression, we can assume beta1 is positive since there's discrimination towards other groups.
Since beta1 is positive, it shows that being white gives a higher likelihood of getting loans approved.
""" 
print(a1)

a2 = """
(ii)Regression results above show that the coefficent for white is 0.2006 and the p value shows that it's statistically significant.
Therefore, it has a significant relatioship with loan approval.
"""
print(a2)
print()

# Handle missing values (example: drop rows with missing values)
df2 = df2.dropna()

X = df2[['white','hrat', 'obrat', 'loanprc', 'unem', 'male', 'married', 'dep', 'sch', 'cosign', 'chist', 'pubrec', 'mortlat1', 'mortlat2', 'vr']]
X = sm.add_constant(X)  # Add a constant term

y = df2['approve']  # Assuming 'white' is the dependent variable

model = sm.OLS(y, X).fit()

print(model.summary())
print()

ans4 = """
(iii) white is still positive and still significant, so yes.
"""
print(ans4)

# Create the interaction term
df2['white_obrat_interaction'] = df2['white'] * df2['obrat']

# Define the independent variables
independent_vars = ['hrat', 'obrat', 'loanprc', 'unem', 'male', 'married', 'dep', 'sch', 'cosign', 'chist', 'pubrec', 'mortlat1', 'mortlat2', 'vr', 'white', 'white_obrat_interaction']

# Create a new DataFrame with the selected variables
X = df2[independent_vars]

# Add a constant term to the independent variables
X = sm.add_constant(X)

# Define the dependent variable
y = df2['approve']

# Fit the OLS (Ordinary Least Squares) model
model = sm.OLS(y, X).fit()

# Print the regression results
print(model.summary())
print()
ans5 = """
(iv)Yes, the interaction is signicant with the p value less than 0.05.
"""
print(ans5)
print()

print("(v)")

# Set the value of obrat to 32
obrat_value = 32

# Extract coefficients from the summary table
coefficients = {
    'const': 1.1858,
    'hrat': 0.0013,
    'obrat': -0.0123,
    'loanprc': -0.1468,
    'unem': -0.0089,
    'male': -0.0018,
    'married': 0.0506,
    'dep': -0.0043,
    'sch': -0.0052,
    'cosign': 0.0257,
    'chist': 0.1263,
    'pubrec': -0.2334,
    'mortlat1': -0.0612,
    'mortlat2': -0.0587,
    'vr': -0.0299,
    'white': -0.1438,
    'white_obrat_interaction': 0.0084,
}

# Create a DataFrame with the values you want to predict for
data_predict = pd.DataFrame(coefficients, index=[0])  # Specify an index

# Update obrat with the desired value
data_predict.at[0, 'obrat'] = obrat_value

# Calculate the predicted probability of approval
predicted_prob = model.predict(data_predict)

# Calculate the marginal effect manually
marginal_effect = model.params['white'] + model.params['white_obrat_interaction'] * obrat_value

# Get the standard errors for the marginal effect
marginal_effect_se = model.bse['white'] + obrat_value * model.bse['white_obrat_interaction']

# Calculate the 95% confidence interval
confidence_interval = (
    marginal_effect - 1.96 * marginal_effect_se,
    marginal_effect + 1.96 * marginal_effect_se
)

print(f'Marginal Effect: {marginal_effect:.4f}')
print(f'95% Confidence Interval: {confidence_interval}')

print()

# Wooldridge Computer Exercise C10.1
print("Wooldridge Computer Exercise C10.1")

# Generate a dummy var
df['post_1979'] = (df['year'] > 1979).astype(int)
print(df.describe())

independent_vars = ['inf', 'rec', 'out', 'def', 'post_1979']
dependent_var = 'i3'

# Create a dummy variable for years after 1979
df['post_1979'] = (df['year'] > 1979).astype(int)

# Select the specified independent and dependent variables
X = df[independent_vars + ['post_1979']]
y = df[dependent_var]

# Add a constant term to the independent variables
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the regression summary
print(model.summary())

ans = """
While the textbook value for the intercept is 1.73, the empirical result is -12.0348. Coefficient for
inf in the textbook is 0.606 and the empirical result is 0.5324. The coefficient for def in the textbook
is 0.513, and the empirical result is 0.0881. There are a lot of differences, so maybe there could be multicollineariyty,
heteroscedasticity and other biases
"""
print(ans)
print()

# Wooldridge Computer Exercise C10.2
print("Wooldridge Computer Exercise C10.2")

# Get the correct names for dependent and independent variables
print(new_df.describe())

# Define the dependent variable and independent variables
y = new_df['chnimp']
X = new_df[['chempi', 'lgas', 'lrtwex', 'befile6', 'affile6', 'afdec6']]

# Add a constant term
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the regression summary
print(model.summary())

ans1 = """
(i)The coefficient for chempi is statistically significant, showing a strong relationship between chemical producation
index and log(chnimp). irtwex has an coefficnet less thn 0.1 but greater than 0.05, so it's marginally significant.
"""

ans2 = """
(ii)Since F-statisitc is significant, so the model
is also significant with joint sigificance.
"""

print(ans1)
print()
print(ans2)

# Define the independent variables and the dependent variable
independent_vars = ['chempi', 'lgas', 'lrtwex', 'befile6', 'affile6', 'afdec6', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
dependent_var = 'chnimp'

# Prepare the independent variables (X) and the dependent variable (y)
X = new_df[independent_vars]
y = new_df[dependent_var]

# Add a constant to the independent variables
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X)
results = model.fit()

# Print regression results
print(results.summary())

ans3 = """
(iii)Based on the regression, none of the dummy variables for months are significant. Since none of their p values
are less than 0.5, the months are not associated with seasonalities. The standard errors and coefficients for all variables changed so much.
"""

print(ans3)
