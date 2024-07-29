import pandas as pd
import api
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from datetime import datetime

# Collect data from the API
data = api.collect_data("BTC")

def process_data(data):
    # Extract and process the data
    prices = data['Data']['Data']
    df =pd.DataFrame.from_dict(prices)[['open', 'time']]


    # Convert 'time' to datetime
    df['time'] = pd.to_datetime(df['time'])

    # Extract additional time-based features
    df['day_of_week'] = df['time'].dt.dayofweek
    df['hour_of_day'] = df['time'].dt.hour
   
    
    # Drop original 'time' column
    df.drop('time', axis=1, inplace=True)

    imputer = SimpleImputer(strategy='median')
    df['open'] = imputer.fit_transform(df[['open']])

    # Define features and target variable
    X = df.drop('open', axis=1)
    y = df['open']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Print predictions and metrics
    print("Predictions:", y_pred[:10])  # Print first 10 predictions for brevity
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

    # Cross-validation score
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    print("Cross-Validation Mean Squared Error:", -cv_scores.mean())

# Process the data
process_data(data)





