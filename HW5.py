import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Chapter 2, Computer Exercise C4 

# Include similar code as HW1 for the basic steps
file_location = "/Users/amyliang/Eco441K/HW5/401ksubs.dta"
f2 = "/Users/amyliang/Eco441K/HW5/stocks.dta"
df1 = pd.read_stata(f2) 
df= pd.read_stata(file_location)

pd.set_option('display.max_columns', None)

# Wooldridge Computer Exercise C4.8
print("Wooldridge Computer Exercise C4.8")

print(df.describe())
print()
print("#1 (i)")

# Filter the dataset for single-person households
single_person_households = df[df['fsize'] == 1]

# Count the number of rows in the filtered dataset
num_single_person_households = single_person_households.shape[0]

# Display the result
print("Number of single-person households is", num_single_person_households)

print()

print("#1(ii)")

# Filter the dataset for single-person households
single_person_data = df[df['fsize'] == 1]

# Define the dependent variable (net financial wealth) and independent variables (income and age)
y = single_person_data['nettfa']
X = single_person_data[['inc', 'age']]

# Add a constant term to the independent variables matrix
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print the regression results
print(model.summary())
ans = """
coefficent for inc indicates that the relatioship between age and net fincial wealth is positive. 
Also, the coefficent for age indicates that there's a positive relationship between age and net financial wealth, which makes sense.
The surprise is the number for intercept, -43.04. We're not sure why it's a negative value, and may need more information to interpret it.
"""
print(ans)

print("#1 (v)")

# Filter the dataset for single-person households
single_person_data = df[df['fsize'] == 1]

# Define the dependent variable (net financial wealth) and independent variable (income)
y_simple = single_person_data['nettfa']
X_simple = single_person_data[['inc']]

# Add a constant term to the independent variable matrix
X_simple = sm.add_constant(X_simple)

# Fit the simple OLS model
model_simple = sm.OLS(y_simple, X_simple).fit()

# Print the regression results for the simple model
print(model_simple.summary())
ans_1 = """
The coefficient for inc in part 2 is 0.7993 and the new inc value is 0.8207. Since the difference is small, then 
additional variable age didn't change the estimated relationship between inc and nettfa. Also, coefficients for
both cases are postive so there's a positive relationship between income and new financial wealth. 
"""

print(ans_1)

print()
print("#1 (iv)")
ans_2 = """
Since p value is les than 0.01, we reject the null hypothesis at 1 percent significance level.
age is statistically siginicant.
"""

print(ans_2)

print()

print("#2 (a)")
print(df1.describe())
# Simple Linear Regression of GE on IBM
model_simple = sm.OLS(df1['ge'], sm.add_constant(df1['ibm'])).fit()
ci_simple = model_simple.conf_int(alpha=0.1)  # 90% confidence interval

# Multiple Linear Regression of GE on IBM and Dow Jones
model_multiple = sm.OLS(df1['ge'], sm.add_constant(df1[['ibm', 'dowjones']])).fit()
ci_multiple = model_multiple.conf_int(alpha=0.1)  # 90% confidence interval

# Display the results
print("Simple Linear Regression - Confidence Interval for IBM Slope:")
print(ci_simple)

print("\nMultiple Linear Regression - Confidence Interval for IBM Slope:")
print(ci_multiple)
 
print()
print("#2 (b)")

# Multiple Linear Regression of GE on IBM and Dow Jones
model_multiple = sm.OLS(df1['ge'], sm.add_constant(df1[['ibm', 'dowjones']])).fit()

# Hypothesis testing for dowjones coefficient
hyp_test_result = model_multiple.t_test('dowjones = 1')

# Display the results
print(hyp_test_result)

ans_3 = """
The null hypothesis means that a one-unit change in the dowjones is equal to 1.
The p value is 0.175
At a 5 percent significance level, the p value of 0.175 is greater than 0.05, thus, we don't have 
enough evidence to reject the null hypothesis.
At a 10 percent level, the p value is still greater than 0.1, thus we fail to reject the null hypothesis
"""

print(ans_3)
