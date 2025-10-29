import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Set a clean plot style
sns.set_style("whitegrid")

def get_gas_price(input_date, model):
    """
    Estimates the natural gas price for a given date using a trained model.

    Args:
        input_date (str or datetime): The date for which to predict the price.
        model (sklearn.linear_model.LinearRegression): The trained regression model.

    Returns:
        float: The predicted price.
    """
    try:
        # 1. Convert input to datetime object
        if not isinstance(input_date, datetime):
            input_date = pd.to_datetime(input_date)
        
        # 2. Perform the same feature engineering as the model
        # 'time_ordinal' captures the long-term linear trend
        time_ordinal = input_date.toordinal()
        
        # 'month_sin' and 'month_cos' capture the 12-month seasonal cycle
        month = input_date.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # 3. Create the feature array (must be 2D for sklearn.predict)
        features = [[time_ordinal, month_sin, month_cos]]
        
        # 4. Predict the price
        predicted_price = model.predict(features)[0]
        
        return predicted_price
    
    except Exception as e:
        return f"Error processing date {input_date}: {e}"

def main():
    """
    Main function to load, train, visualize, and demonstrate the model.
    """
    # --- 1. Load and Inspect Data ---
    file_name = 'Nat_Gas.csv'
    try:
        df = pd.read_csv(file_name, parse_dates=['Dates'])
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        print("Please make sure 'Nat_Gas.csv' is in the same directory as the script.")
        return

    df = df.sort_values('Dates').reset_index(drop=True)
    print("Data loaded successfully. Head:")
    print(df.head())
    print("\nData Info:")
    df.info()

    # --- 2. Feature Engineering (for Training) ---
    # To model seasonality, we can't just use 'month' as a number (1-12),
    # as this implies month 12 is 12x "stronger" than month 1.
    # Instead, we use sine and cosine to represent the cyclical nature.
    
    # 'time_ordinal' captures the long-term linear trend
    df['time_ordinal'] = df['Dates'].apply(lambda x: x.toordinal())
    
    # 'month_sin' and 'month_cos' capture the 12-month seasonal cycle
    df['month_sin'] = np.sin(2 * np.pi * df['Dates'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Dates'].dt.month / 12)

    # --- 3. Train the Model ---
    # We will model Price as a function of:
    # Price = Intercept + (slope * time_ordinal) + (amp1 * month_sin) + (amp2 * month_cos)
    
    features = ['time_ordinal', 'month_sin', 'month_cos']
    X_train = df[features]
    y_train = df['Prices']

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nModel trained successfully.")

    # --- 4. Generate Dates for Plotting (Interpolation & Extrapolation) ---
    # We want a daily price, so we create a full date range from
    # the start of our data to one year past the end.
    
    start_date = df['Dates'].min()
    # Data ends 2024-09-30, so one year future is 2025-09-30
    end_date = df['Dates'].max() + pd.DateOffset(years=1)
    
    # Create a daily date range
    plot_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    plot_df = pd.DataFrame({'Dates': plot_dates})

    # --- 5. Predict on Plotting Dates ---
    # Apply the same feature engineering to the new daily dates
    plot_df['time_ordinal'] = plot_df['Dates'].apply(lambda x: x.toordinal())
    plot_df['month_sin'] = np.sin(2 * np.pi * plot_df['Dates'].dt.month / 12)
    plot_df['month_cos'] = np.cos(2 * np.pi * plot_df['Dates'].dt.month / 12)
    
    # Predict prices for this entire daily range
    X_plot = plot_df[features]
    plot_df['Predicted_Price'] = model.predict(X_plot)

    # --- 6. Visualize the Results ---
    plt.figure(figsize=(14, 8))
    
    # Plot the modeled/forecasted line
    plt.plot(plot_df['Dates'], plot_df['Predicted_Price'], 
             label='Modeled Price (Daily)', color='red', linestyle='--')
    
    # Plot the original monthly data points
    plt.scatter(df['Dates'], df['Prices'], 
                label='Original Monthly Data', color='blue', zorder=5)
    
    # Add a vertical line to show where the forecast begins
    forecast_start_date = df['Dates'].max()
    plt.axvline(x=forecast_start_date, color='green', linestyle=':', 
                label=f'Forecast Start ({forecast_start_date.date()})')

    plt.title('Natural Gas Price: Historical, Modeled, & Forecasted', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    
    # Save the plot
    plot_filename = 'nat_gas_price_forecast.png'
    plt.savefig(plot_filename)
    print(f"\nSuccessfully saved plot to '{plot_filename}'")

    # --- 7. Demonstrate the Function ---
    print("\n--- Price Prediction Examples ---")
    
    # Example 1: Interpolation (a date between known points)
    date_1 = '2024-05-15'
    price_1 = get_gas_price(date_1, model)
    print(f"Predicted price for {date_1}: ${price_1:.2f}")

    # Example 2: Extrapolation (a date in the future)
    date_2 = '2025-01-15'
    price_2 = get_gas_price(date_2, model)
    print(f"Predicted price for {date_2}: ${price_2:.2f}")

    # Example 3: Extrapolation (a future summer date)
    date_3 = '2025-07-01'
    price_3 = get_gas_price(date_3, model)
    print(f"Predicted price for {date_3}: ${price_3:.2f}")

if __name__ == "__main__":
    main()