import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Loading the dataset
path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(path)

# Displaying the first five rows and statistics
print("\n")
print("first rows of the dataset: ")
print("\n")
print(df.head())
print("\n")
print("calculated statistics: ")
print("\n")
print(df.describe())

# Selecting relevant features
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Visualizng the data with histograms 
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.suptitle("Histograms") 
plt.show()


plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Fuel consumption")
plt.ylabel("Emission")
plt.title("Fuel consumption vs CO2 emission") 
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Engine size vs CO2 emission") 
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.title("No. of cylinders vs CO2 emission") 
plt.show()

# Splitting data into training and test sets
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Training the model


from sklearn import linear_model
regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# Printing the coefficients and intercept
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

from sklearn.metrics import r2_score

# Testing the model
test_x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y, test_y_))

# Prediction function
def predict_co2(engine_size, cylinders, fuel_consumption):
    input_features = np.array([[engine_size, cylinders, fuel_consumption]])
    predicted_co2 = regr.predict(input_features)
    return predicted_co2[0][0]

print("\n")
input_engine_size = float(input("Enter the engine size to predict CO2 emissions (L): "))
input_cylinders = int(input("Enter the number of cylinders: "))
input_fuel_consumption = float(input("Enter the fuel consumption (L/100 km): "))
print("\n")
predicted_co2 = predict_co2(input_engine_size, input_cylinders, input_fuel_consumption)
print(f"Predicted CO2 emissions for engine size {input_engine_size}, cylinders {input_cylinders}, and fuel consumption {input_fuel_consumption}: {predicted_co2:.2f}")