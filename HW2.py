import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Chapter 2, Computer Exercise C4 

# Include similar code as HW1 for the basic steps
file_location = "/Users/amyliang/Eco441K/HW2/WAGE2.DTA"
df= pd.read_stata(file_location)

# Wooldridge Computer Exercise C2.4 (parts (i) and (ii))
print("Wooldridge Computer Exercise C2.4 (parts (i) and (ii))")

print("Question (i)")

# Find Avg salary
avg_salary = round(df['wage'].mean(), 3)
avg_iq = round(df['IQ'].mean(), 3)
avg_std_iq = round(df['IQ'].std(), 3)

print("The average salary is", avg_salary)
print("The average IQ is", avg_iq)
print("The sample standard deviation for IQ is", avg_std_iq)

print()

print("Question (ii)")

# List out the independent var (IQ) and dependent var (wage)
X = df['IQ']
y = df['wage']

# Add a constant term for the intercept
X = sm.add_constant(X)

# Fit the simple linear regression model
model = sm.OLS(y, X).fit()

# Print the summary statistics of the regression model
print(model.summary())
print()
# Increase IQ by 15 points and see if it increases wage
increase_iq = 15
change_in_wage = round(model.params['IQ'] * increase_iq, 3)
print("The predicated increase in wage for the increase in IQ is", change_in_wage)

rsquared_value = round(model.rsquared, 3)

print("The rsquared value is", rsquared_value, "meaning that IQ doesn't explain the variance in the wage.")

#
# Remaining Computer Questions
print()
print("Remaining Computer Questions")
print()
print("Question 1")

# Identify the data needed
iq = df['IQ']
wage = df['wage']

# Create a scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(iq, wage, color = 'hotpink', label='Individual data')

# Add a simple linear regression model
X = sm.add_constant(iq)  # Add a constant for the intercept
model = sm.OLS(wage, X).fit()

# Generate predicted values based on the fitted model
iq_range = np.linspace(iq.min(), iq.max(), 100)
iq_const = sm.add_constant(iq_range) # Add an iq constant range
predicted_wages = model.predict(iq_const)

# Plot the regression line
plt.plot(iq_range, predicted_wages, color='blue', linewidth=2, label='Fitted Regression Line')

# Set plot labels and title
plt.xlabel('IQ')
plt.ylabel('Wage')
plt.title('Wage vs IQ with Fitted Regression Line')

# Add a legend
plt.legend()

# Show the plot
plt.show()

# Question 2
print("Question 2")

# Part 1
# Calculate covariance between wage and IQ
cov_wage_iq = df['wage'].cov(df['IQ'])

# Calculate variance of IQ
var_iq = df['IQ'].var()

# Calculate the slope estimate
slope_ratio_estimate = round(cov_wage_iq / var_iq, 3)
print("The ratio between the sample covariance (between wage and iq) and the variance of iq is", slope_ratio_estimate)

# Part 2
# Calculate sample correlation between wage and IQ
corr_wage_iq = df['wage'].corr(df['IQ'])

# Calculate standard deviation of wage and IQ
std_wage = df['wage'].std()
std_iq = df['IQ'].std()

# Calculate the second ratio
sec_slope_ratio = round((std_wage / std_iq) * corr_wage_iq, 3)
print("The sample correlation (between wage and iq) times the ratio of the standard deviations of the two variables is", sec_slope_ratio)

# Test if they are equal
if slope_ratio_estimate == sec_slope_ratio:
    print("The regression slope estimate is", slope_ratio_estimate, "and the two numbers match.")
else:
    print("Part i is," + str(slope_ratio_estimate) + ", while part ii is", sec_slope_ratio, "thus, they're different.")

# Question 3
print()
print("Question 3")

# List out the independent var (IQ) and dependant var (wage)
X = df['IQ']
y = df['wage']

# Add a constant for the intercept
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Use python's predict function and get fitted values for wage or (wagehat)
wagehat = model.predict(X)

# Calculate the sample average of wage and wagehat
sample_avg_wage = round(y.mean(), 3)
sample_avg_wagehat = round(wagehat.mean(), 3)

if sample_avg_wage == sample_avg_wagehat:
    print("The sample average of wagehat and the sample average of wage are the same", sample_avg_wage)
else:
    print("The sample average of wagehat is", sample_avg_wagehat, "while the sample average of wage is", sample_avg_wage)

# Calculate the correlation between wagehat and IQ
corr_wagehat_iq = pd.Series(wagehat).corr(df['IQ'])
print("The corr between wagehat and IQ is", corr_wagehat_iq)
ans_3 = """
Even without using coding, we can tell that the average wagehat and the mean of IQ are very close,
and the numbers are positive, so we can assume the correlation is positive")
"""
print(ans_3)
print()

# Question 4
print("Question 4")

print("Part 1")

# Calculate the OLS residuals (uhat)
uhat = y - wagehat

# Verify if the sample average of uhat is equal to zero
sample_avg_uhat = round(uhat.mean(), 3)
if sample_avg_uhat != 0:
    print("the sample average of uhat is", sample_avg_uhat)
else:
    print("the sample average of uhat is 0")

print()
print("Part 2")

# Verify if uhat and iq's correclation is 0
corr_uhat_iq = round(pd.Series(uhat).corr(df['IQ']), 3)
if corr_uhat_iq != 0:
    print("The correlation between uhat and IQ is", corr_uhat_iq)
else:
    print("The correlation between uhat and IQ is 0")







