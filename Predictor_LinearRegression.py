import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

""" Goal: Predict the general price of homes in the largest cities across Canada 
    based on different attributes. The attributes used are all the columns with 
    the exception of the price (Desired output) and the addresses (Noisy feature).

    The link to the dataset is here: 
    https://www.kaggle.com/datasets/jeremylarcher/canadian-house-prices-for-top-cities
"""


"""#--- MODEL 1: LINEAR REGRESSION
    We will use a linear regression model to predict the price of homes based on the attributes
    The model will be evaluated using the Root Mean Squared Error (RMSE) and R^2 score to track error margins.
    Model accuracy will be improved by removing outliers from the dataset beyond 3 standard deviations.
"""
# Load the dataset from a CSV file and drop the address column (too noisy) and NaN values
data = pd.read_csv('HouseListing-Top45.csv', encoding='latin1')
data = data.drop(labels='Address', axis=1)
data.dropna(inplace=True)
data.info()

#Use one-hot encoding to convert categorical variables (City, Province) to numerical
data = pd.get_dummies(data, columns=['City', 'Province'], drop_first=True)

#Clean the data by removing outliers using the Z-score method
zScore = np.abs((data.iloc[:, 0] - data.iloc[:, 0].mean()) / data.iloc[:, 0].std())

#Get the train and test sets for the model (80% train, 20% test)
#We remove the address and price columns from the training attributes
#We only keep the price column for the output (y values)
xTrain, xTest, yTrain, yTest = train_test_split(data.iloc[:, 1:], 
                                                data.iloc[:, 0], 
                                                test_size=0.2, 
                                                random_state=42)

#Constrain data to 3 standard deviations from the mean to remove crazy outliers
xTrain = xTrain[zScore < 3]
xTest = xTest[zScore < 3]
yTrain = yTrain[zScore < 3]
yTest = yTest[zScore < 3]

#Create and train the first model (linear model)
linearRegModel = LinearRegression().fit(xTrain, yTrain)

#Make predictions on the test set
yPred = linearRegModel.predict(xTest)

#Now that the model is trained we can factor in the margins of error
#10% margin of error for the linear model
linearMargin = 0.1

linearLowerBound = yPred * (1 - linearMargin)
linearUpperBound = yPred * (1 + linearMargin)

#Calculate the error metrics
rmse = root_mean_squared_error(yTest, yPred)
r2 = r2_score(yTest, yPred)
print(f'Root Mean Squared Error: {rmse:.2f}\nR^2 Score: {r2:.2f}')


#--- Visualization of results ---

#On a scatter plot, mark the actual outcomes of the test set
#Scale down the prices by a factor of 1 million
yTestScaled = yTest/1e7
yPredScaled = yPred/1e7

#Mark the predicted values of the test set on a line
plt.scatter(yTestScaled, yPredScaled, marker='o', color='blue', label='Actual Prices')
coeff = np.polyfit(yTestScaled, yPredScaled, 1)
lobf = np.polyval(coeff, yTestScaled)
#Plot the line of best fit from the predicted values
plt.plot(yTestScaled, lobf, color='red', linewidth=2, label='Predicted Prices')
#Plot the margins of error
plt.plot(yTestScaled, 
        lobf*(1-linearMargin), 
        linestyle='dashed', color='red', alpha=0.5, 
        label=str(linearMargin*100)+'% Margin of Error (Lower Bound)')
plt.plot(yTestScaled,
        lobf*(1+linearMargin),
        linestyle='dashed', color='red', alpha=0.5, 
        label=str(linearMargin*100)+'% Margin of Error (Upper Bound)')
#Label the x-axis, y-axis, and the plot
plt.xlabel('Actual Price (millions $)')
plt.ylabel('Predicted Price (millions $)')
plt.title('House Price Prediction')
plt.legend()
plt.show()

