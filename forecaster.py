from data_loader import load_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def main():
    # Load data using the helper function
    X_train, X_test, y_train, y_test = load_data()

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"Model Training Complete.")
    print(f"Mean Absolute Error: {mae:.2f}")

if __name__ == "__main__":
    main()