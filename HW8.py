import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Include similar code as HW1 for the basic steps
file_location = "/Users/amyliang/Eco441K/HW8/VOTE1.DTA"
df= pd.read_stata(file_location)
f2 = "/Users/amyliang/Eco441K/HW8/nbasal.dta"
df2 = pd.read_stata(f2)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Wooldridge Computer Exercise C6
print("Wooldridge Computer Exercise C6")

# Define the independent variables (explanatory variables)
X = df[['prtystrA', 'expendA', 'expendB', 'shareA']]

# Add a constant term (intercept) to the model
X = sm.add_constant(X)

# Define the dependent variable
Y = df['voteA']

# Fit the linear regression model
model = sm.OLS(Y, X).fit()

# Print the regression results summary
print(model.summary())
ans1 = """
(i)As the coefficient for 'expend' directly isn't present in the output, we can't directly interpret its isolated effect.
Regarding the expected sign for B4, typically, the sign for an interaction term
like 'expendA-expend' (which is represented by B4 in the model) is not immediately obvious without further context or understanding of the variables involved. 
The interaction term often signifies the joint effect of 'expendA' and 'expendB' and can't be independently interpreted without considering the individual 
variables involved in the interaction. 
"""
ans = """
(ii)The variables 'prtystrA' and 'shareA' appear to be statistically significant in predicting 'voteA'.
However, 'expendA' and 'expendB' might need further scrutiny due to their lower significance levels or potential multicollinearity issues
with other variables.
"""
print(ans1)
print(ans)

ans2 = """
(iii)Whether this effect is considered large or not would depend on the context and the scale of the variable 'voteA'. 
A coefficient of 0.0043 implies that for every $100,000 increase in Candidate B's expenditure,
voteA would be expected to change by 0.43 votes, given other variables remain constant. 
The significance of this effect would depend on the domain and the significance of a single vote in the context of the scenario.
"""
print(ans2)

ans3 = """
Effect = $100 * coefficient for 'expendA'

This value represents the expected change in 'voteA' for every $100 increase in 'expendA', given 'expend' is fixed at $100.
This maybe reasonable depending on the cases or scenarios given.
"""

print(ans3)

print(df2.describe())
# Create the quadratic term for 'Experience'
df2['expersq'] = df2['exper'] ** 2

# Define the dependent variable (Points)
Y = df2['points']

# Define the independent variables including Experience and Experience_Squared
X = df2[['exper', 'expersq']]

# Add a constant term (intercept) to the model
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(Y, X).fit()

# Print the regression results summary
print(model.summary())
print()

ans4 = """
(i) he overall model with experience and its quadratic term appears to have some explanatory power for the points variable. 
The individual effects of exper and expersq are statistically suggestive but might not be highly significant,
given the higher p-values.It seems that the quadratic relationship between experience and points scored
might provide some added explanation beyond a linear relationship, but the evidence is not overwhelmingly strong
"""
print(ans4)

ans5 = """
(ii)The absence of including all three position dummy variables might be due to a choice made in the analysis to use only two of the three position dummies as reference groups. 
we want to avoid multicollinearity.
"""
print(ans5)

ans6 = """
(iii)To compare guards and centers, a statistical test could be performed on the coefficient of the guard dummy variable to ascertain whether guards 
score significantly more points than centers, holding experience fixed.
"""
print(ans6)

ans7 = """
(iv)Adding marital status to the equation allows for an examination of the productivity difference between married and unmarried players while controlling for position and experience.
A statistical test can determine if marital status significantly affects points per game.
"""
print(ans7)

ans8= """
(v)Adding interactions between marital status and experience variables provides insight into whether being married influences how experience relates to points per game.
Statistical tests can ascertain if these interactions have a significant effect on points per game.
"""
print(ans8)
# Define the dependent variable (Assists per Game)
Y = df2['assists']

# Define the independent variables (including marital status, position, experience, and their interactions)
X = df2[['exper', 'expersq', 'guard', 'forward', 'marr']]

# Add a constant term (intercept) to the model
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(Y, X).fit()

# Print the regression results summary
print(model.summary())

print()

ans9 = """
(vi)With the given data, he guard position seems to significantly influence the number of assists, 
while forward players have a less influential but still notable impact. Marital status and experience, including its quadratic form, 
do not seem to significantly affect assists per game in this model. 
"""

print(ans9)
