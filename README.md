# HomePricePredictior_v1.1
This linear regression model uses a dataset from Kaggle (https://www.kaggle.com/datasets/jeremylarcher/canadian-house-prices-for-top-cities) containing more instances (35768) and more attributes (8) compared to the previous dataset. This model crops outliers from beyond 3 standard deviations and detects the error margins on the trained model. The outcome is plotted through MatPlotLib and can be found in Price_Prediction_v1.1.png.

# Features
- Splits the data into training and test sets (80/20 ratio)
- Trains a linear regression model using scikit-learn
- Predicts house prices from the test set
- Visualizes actual vs. predicted prices using matplotlib
  - **Blue dots**: Actual house prices from the test data
  - **Red line**: Predicted prices from the linear regression model

# HomePricePredictor_v1
This prototype simple linear regression model demonstrates the use of scikit-learn to predict house prices based on the size of the house. This is done by using a simple CSV dataset sourced from Zillow,  containing the house's size in square footage and its corresponding prices in millions of dollars. The outcome will be found on Price_Prediction_v1.png.



  # Dependencies

  Install the following libraries using pip:

  ```bash
  pip install numpy pandas matplotlib scikit-learn

