import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Chapter 2, Computer Exercise C4 

# Include similar code as HW1 for the basic steps
file_location = "/Users/amyliang/Eco441K/HW3/MEAP93.DTA"
df= pd.read_stata(file_location)

pd.set_option('display.max_columns', None)

# Wooldridge Computer Exercise C2.6
print("Wooldridge Computer Exercise C2.6")

print(df.describe())

# Scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(df['expend'], df['gradrate'])
plt.title('Scatterplot of Expenditure vs. Graduation Rate')
plt.xlabel('Expenditure')
plt.ylabel('Graduation Rate')

# Linear regression
X = df['expend']
X = sm.add_constant(X)
y = df['gradrate']
model = sm.OLS(y, X).fit()

# Plotting the regression line
predicted = model.predict(X)
plt.plot(X['expend'], predicted, color='red', linewidth=2, label='Regression Line')

# Show the plot
plt.legend()
plt.show()

# Display regression summary
print(model.summary())
ans_1 = """
based on the OLS Regression result, the coefficient is very small, the p value is not statistically significant, and the R-squared
is 0.001, indicating that the model explains little variations to the graduation rate. The data doesn't
fit well. suggests that other factors, not included in the model, may have a more substantial influence on graduation rates.
"""
print("(i)", ans_1)

ans_2 = """
If beta_1 represents the percentage change in math10 for a 1 percent change in then 
beta_1 / 10 would represent the percentage change in math10 for a 10 percent change in expend.
Afterall, a percent change iis equivalent to ten 1 percent changes.
"""
print("(ii)",ans_2)

# Part iii
print("(iii)")
# List out the independent and dependent var
X = df['lexpend']  # Independent variable (log of expend)
y = df['math10']   # Dependent variable (math10)

# Add a constant to the independent variable
X = sm.add_constant(X)

# Fit the OLS regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression results
print(model.summary())

ans_3 = """
Based on the data, our model is still math10 = beta0 + beta1 * log(expend) + u. The sample
size is 408, the R-squared is 0.030 , showing that there's 3 percent of the variation in math10
that is explained by log(expend).
The model shows that there's a 10 percent increase in expend (which corresponds to an increase in log(expend) by 1) 
that is associated with an increase of approximately 11.1644 units in math10, holding other factors constant.
"""
print(ans_3)

answ_4 = """
The estimated coefficient for log(expend) in the model is approximately 11.1644.
Percentage Point Increase in math10 = (Coefficient for log(expend)) * 10
                                    = 11.1644 * 10 = 111.644 (in percentage)
"""
print("(iv)", answ_4)

ans_5 = """
First, the scores in math10 are not highly skewed once the values are above 100. Second, the estimated model may not fit the data perfectly,
and there may be variability in the relationship between spending and math10 that can prevent the model
from predicting values that are unrealistically high.
"""

print("(v)", ans_5)

# Part vi
print("(vi)")
# List out the var
math10 = df['math10']   
lexpend = df['lexpend']  

# Fit the regression model
X = sm.add_constant(lexpend)
model = sm.OLS(math10, X).fit()
intercept, slope = model.params

# Create a scatter plot
plt.scatter(lexpend, math10, color='hotpink', label='Individual data')

# Add the fitted regression line
plt.plot(lexpend, intercept + slope * np.array(lexpend), color='blue', label='Fitted Line')

# Labeling and legend
plt.xlabel('log(expend)')
plt.ylabel('math10')
plt.legend()

# Show the plot
plt.show()

# Part vii
print("(vii)")

# Your data for expend and lexpend
expend = df['expend']  # Replace with your actual data
lexpend = df['lexpend']  # Replace with your actual data

# Fit the regression model
X = sm.add_constant(lexpend)
model = sm.OLS(math10, X).fit()
fitted_values = model.fittedvalues

# Create a scatter plot
plt.scatter(expend, fitted_values, label='Fitted Values', alpha=0.5, color='hotpink')

# Labeling and legend
plt.xlabel('expend')
plt.ylabel('Fitted Values')
plt.legend()

# Show the plot
plt.show()
ans_add = """
As expenditures continue to increase, the slope of the scatter plot starts to flatten or decrease. Thus,
spending more money beyond a certain point does not lead to a proportionate increase in math10 scores.
The scatter plot indicates a non-linear relationship between expenditure and math10 scores.
While there is an initial positive impact of spending more on education, there seems to be a point of diminishing returns
"""
print(ans_add)

print("viii")

# Re-scale math10 by dividing by 100
df['math10'] = df['math10'] / 100

# Regression with re-scaled math10
X_rescaled = sm.add_constant(df['lexpend'])
y_rescaled = df['math10']

model_rescaled = sm.OLS(y_rescaled, X_rescaled).fit()

print(model_rescaled.summary())

ans_6 = """
R-squared and SST won't change because they measure the variability of the dependent variable by the independent
variable. However, the slope estimate and intercept estimate will change once we re-scale the equation. 
"""
print(ans_6)
















