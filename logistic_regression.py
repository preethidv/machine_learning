from sklearn.linear_model import LogisticRegression
import numpy

# logistic regression is a classification algorithm. The results are true/false, 0/1 basically, binomial values.
# In this example, we take the inputs (X) as size of the tumor, and outputs (y) as whether it is cancerous or not.

# the size of a tumor in centimeters.
#Note: X has to be reshaped into a column from a row for the LogisticRegression() function to work.
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)

#y represents whether or not the tumor is cancerous (0 for "No", 1 for "Yes").
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression();
model.fit(X,y)

predicted = logr.predict(numpy.array([3.46]).reshape(-1,1))
print("Prediction whether size 3.46 cms of tumor is cancerous:"+predicted)

# In logistic regression the coefficient is the expected change in log-odds of having the outcome per unit change in X.
# Here, the coefficient will indicate what is the odds of being cancerous for every 1cm increase in size of the tumor.
log_odds = model.coef_
odds = numpy.exp(log_odds)
print(odds)  #This tells us that as the size of a tumor increases by 1cm the odds of it being a cancerous tumor increases by 4x.

# The coefficient and intercept values can be used to find the probability that each tumor is cancerous.
# Create a function that uses the model's coefficient and intercept values to return a new value. This new value represents probability 
# that the given observation is a tumor

def probabFunc(model,x):
  log_odds = model.coef_ * x + model.intercept_
  odds = numpy.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

print(probabFunc(model,X))

# Answers explained:
# 3.78 0.61 The probability that a tumor with the size 3.78cm is cancerous is 61%.
# 2.44 0.19 The probability that a tumor with the size 2.44cm is cancerous is 19%.
# 2.09 0.13 The probability that a tumor with the size 2.09cm is cancerous is 13%.
      


