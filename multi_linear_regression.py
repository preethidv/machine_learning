import pandas as pd
from sklearn.linear_model import LinearRegression

data_set = pd.read_csv("data\\CarbonEmission.csv")
X = data_set[['Weight','Volume']]
y = data_set[['CO2']]
model = LinearRegression()
model.fit(X, y)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = model.predict([[2300, 1300]]);
print(predictedCO2)

# to get the coefficients -- basically, to understand how much CO2 will increase for each
# 1 kg increase in weight and 1cm3 increase in volume
print(model.coef_)
