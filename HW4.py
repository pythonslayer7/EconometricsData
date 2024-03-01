import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Include similar code as HW1 for the basic steps
file_location = "/Users/amyliang/Eco441K/HW4/attend.dta"
f = "/Users/amyliang/Eco441K/HW4/attend-1.dta"
df2 = pd.read_stata(f)
df= pd.read_stata(file_location)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Wooldridge Computer Exercise C3.4
print("Wooldridge Computer Exercise C3.4")

print(df2.describe())
print(df.describe())


print("(i)")
print("The min of atndrte is", df['atndrte'].min())
print("The max of atndrte is", df['atndrte'].max())
print("The mean of atndrre is", df['atndrte'].mean())

print("The min of priGPA is", df['priGPA'].min())
print("The max of priGPA is", df['priGPA'].max())
print("The mean of priGPA is", df['priGPA'].mean())

print("The min of ACT is", df['ACT'].min())
print("The max of ACT is", df['ACT'].max())
print("The mean of ACT is", df['ACT'].mean())
print()
print("(ii)")
ans_2 = """
The equation form is Attendance Rate = Beta_0 + Beta_1 * Prior GPA + Beta_2 * ACT + Error Term

The intercept represents the estimated attendance rate when both Prior GPA and ACT are 0. But this is not meaningful
because students in real life won't have a Prior GPA and ACT scores of 0. 
"""
print(ans_2)

X = df2[['priGPA', 'ACT']]  # Independent variables
X = sm.add_constant(X)  # Add a constant (intercept) term
Y = df2['termgpa']  # Dependent variable

# Fit the linear regression model
model = sm.OLS(Y, X).fit()

# Get the coefficient estimates
beta_0_estimate = model.params['const']  # Intercept estimate
beta_1_estimate = model.params['priGPA']  # Coefficient for priGPA
beta_2_estimate = model.params['ACT']  # Coefficient for ACT

print("beta_0 estimate is", beta_0_estimate)
print("beta_1 estimate is", beta_1_estimate)
print("beta_2 estimate is", beta_2_estimate)

ans_3 = """
Based on the simple regression model, we see that the estimated slope of Prior GPA and ACT are both positively correlated
with attendance rate, which is expected. However, beta_2 estimate is very low, almost close to 0. This is suprising. 
"""
print(ans_3)

print()

print("(iv)")
ans_4 = """
The predicted atndrte is about 3.52, this means that the attendance rate for a student this given priGPA and ACT score is 3.52.
There are students with priGPA and ACT scores close to the given number, and the attendance rate is about 3.6,
which is close to this calculation. 
"""
print(ans_4)

print()
print("(v)")
ans_5 = """Once we plug in the values, student A has about attendance rate of 3.0 while student B has attendance rate about
2.19. after subtraction, we get a difference for about 0.81. Therefore, student has higher attendance rate for about 0.81 than student B.
"""
print(ans_5)
