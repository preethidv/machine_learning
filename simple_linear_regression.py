import numpy as np
import pandas as pd
import sklearn.model_selection import train_test_split
import sklearn.linear_model import LinearRegression
import matplotlib.pyplot as mtp

data_set = pd.read_csv("data\\Salary_Data.csv")
print(data_set)

x = data_set.iloc[:,:-1].values # extract the first column, exclude last
y = data_set.iloc[:,1].values # get the last column data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)
x_pred = model.predict(x_train)
y_pred = model.predict(x_test)

# visualize the training data
mtp.scatter(x_train, y_train, color = "green")
mtp.plot(x_train, x_pred, color="blue")
mtp.xlabel("Years of experience)
mtp.ylabel("Salary")
mtp.title("Salary vs Years of Experience (Training Set")
mtp.show()

# visualize the test data
mtp.scatter(x_test, y_test, color = "red")
mtp.plot(x_train, x_pred, color="blue")
mtp.xlabel("Years of experience)
mtp.ylabel("Salary")
mtp.title("Salary vs Years of Experience (Test Set")
mtp.show()

# Now, can predict y for a given X, or a set of X values
y_pred = model.predict(5)
      
