import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f


# Include similar code as HW1 for the basic steps
file_location = "/Users/amyliang/Eco441K/HW6/LAWSCH85.DTA"
df= pd.read_stata(file_location)

pd.set_option('display.max_columns', None)

# Wooldridge Computer Exercise C4.2
print("Wooldridge Computer Exercise C4.2")
print(df.describe())

print("(i)")
# Define the dependent variable (y) and the independent variable (X)
y = df['salary']
X = sm.add_constant(df['rank'])  # Add a constant term for the intercept

# Fit the regression model
model = sm.OLS(y, X).fit()

# Perform the hypothesis test on the coefficient for 'rank'
hypothesis = 'rank = 0'
t_test = model.t_test(hypothesis)

# Extract the p-value for 'rank'
p_value = t_test.effect[0]

# Set the significance level (e.g., 0.05)
alpha = 0.05

# Check if the p-value is less than the significance level
if p_value < alpha:
    print(f" We reject the null hypothesis. The rank has a ceteris paribus effect on median starting salary.")
else:
    print(" We fail to reject the null hypothesis. The rank does not have a ceteris paribus effect on median starting salary.")

print()
print("(ii)")
# Remove rows with missing values in 'salary', 'LSAT', and 'GPA' columns
df_clean = df.dropna(subset=['salary', 'LSAT', 'GPA'])



# Define the dependent variable (y) and the independent variables (X)
y = df_clean['salary']
X = df_clean[['LSAT', 'GPA']]
X = sm.add_constant(X)  # Add a constant term for the intercept

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Perform the hypothesis tests on the coefficients for 'LSAT' and 'GPA'
t_test_lsat = model.t_test('LSAT = 0')
t_test_gpa = model.t_test('GPA = 0')

# Extract the p-values for 'LSAT' and 'GPA'
p_value_lsat = t_test_lsat.pvalue
p_value_gpa = t_test_gpa.pvalue
print("The p value for LSAT is", p_value_lsat)
print("The p value for GPA is", p_value_gpa)

# Set the significance level (e.g., 0.05)
alpha = 0.05

# Check if the p-values are less than the significance level
if p_value_lsat < alpha and p_value_gpa < alpha:
    print("Both LSAT and GPA are individually significant for explaining salary.")
elif p_value_lsat < alpha or p_value_gpa < alpha:
    print("At least one of LSAT and GPA is individually significant for explaining salary.")
else:
    print("Neither LSAT nor GPA is individually significant for explaining salary.")

print()
print("(iii)")

# Remove rows with missing values in 'salary', 'LSAT', 'GPA', 'clsize', and 'faculty' columns
new_df = df.dropna(subset=['salary', 'LSAT', 'GPA', 'clsize', 'faculty'])

# Define the dependent variable (y) and the initial independent variables (LSAT and GPA)
y = new_df['salary']
X_initial = new_df[['LSAT', 'GPA']]
X_initial = sm.add_constant(X_initial)  # Add a constant term for the intercept

# Fit the initial multiple linear regression model
model_initial = sm.OLS(y, X_initial).fit()

# Define the additional independent variables (clsize and faculty)
X_additional = new_df[['clsize', 'faculty']]
X_additional = sm.add_constant(X_additional)  # Add a constant term for the intercept

# Fit the extended multiple linear regression model
model_extended = sm.OLS(y, X_additional).fit()

# Perform the F-test to compare the fit of the two models
f_statistic = (model_initial.ssr - model_extended.ssr) / (model_extended.df_model - model_initial.df_model) / model_extended.ssr / (model_extended.nobs - model_extended.df_model)

# Calculate the critical F-value for your desired significance level (e.g., 0.05)
from scipy.stats import f
alpha = 0.05
dfn = model_extended.df_model - model_initial.df_model
dfd = model_extended.nobs - model_extended.df_model
critical_f_value = f.ppf(1 - alpha, dfn, dfd)

# Check if the F-statistic is greater than the critical F-value
if f_statistic > critical_f_value:
    print("The size of the entering class (clsize) or the size of the faculty (faculty) should be added to the equation.")
else:
    print("There is no need to add clsize and faculty to the equation.")

print()
print("(iv)")
ans = """
There could factors such as funding for faculty and students, as well as research quantity and quality from the school's faculty.
The passing rate for bar exam in the school can also impact the rank.
"""
print(ans)
 



