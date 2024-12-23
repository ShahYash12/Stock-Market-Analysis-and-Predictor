# Stock Market Analysis and Prediction using LSTM 📈

## Project Overview
This project explores stock market data to analyze and predict stock prices using a Long Short-Term Memory (LSTM) model. By utilizing Python libraries and techniques, we delve into various financial metrics, perform data visualization, and attempt to predict the closing price of Apple Inc.'s stock.

## Objectives
1. **Analyze stock market data:**
   - Track the change in stock prices over time.
   - Calculate the daily return of stocks on average.
   - Compute the moving average of stock prices.
   - Assess the correlation between different stocks.
   - Measure the risk associated with investing in a specific stock.

2. **Predict stock behavior:**
   - Use an LSTM model to forecast the closing price of Apple Inc.'s stock.

---

## Methodology

### Data Collection
- **Source:** Yahoo Finance
- **Tool:** [yfinance](https://pypi.org/project/yfinance/) library
- **Process:** Loaded historical stock market data for analysis and modeling.

### Data Analysis

#### 1. **Change in Stock Prices Over Time**
- **Goal:** Visualize the stock price trend over time.
- **Method:**
  - Extracted the `Close` prices for selected stocks.
  - Used Matplotlib and Seaborn for line plots.

#### 2. **Daily Return of the Stock on Average**
- **Goal:** Calculate the mean daily return of stocks.
- **Method:**
  - Computed daily returns using the formula:
    ```
    daily_return = (price_today - price_yesterday) / price_yesterday
    ```
  - Visualized returns using histograms and box plots.

#### 3. **Moving Average of Stocks**
- **Goal:** Smooth out short-term fluctuations to observe trends.
- **Method:**
  - Calculated moving averages (e.g., 20-day, 50-day) using Pandas rolling mean.
  - Overlayed moving averages on stock price plots.

#### 4. **Correlation Between Stocks**
- **Goal:** Determine the relationship between stock price movements.
- **Method:**
  - Computed correlation matrix using Pandas.
  - Visualized correlations with heatmaps.

#### 5. **Value at Risk (VaR)**
- **Goal:** Quantify the risk of investment loss.
- **Method:**
  - Used Monte Carlo simulations and historical methods to estimate VaR.
  - Analyzed risk at different confidence intervals (e.g., 95%).

### Stock Price Prediction with LSTM
- **Goal:** Predict Apple Inc.'s closing price.
- **Method:**
  1. **Data Preprocessing:**
     - Scaled data using MinMaxScaler to normalize values.
     - Created sequences of stock prices for training the model.
  2. **Model Building:**
     - Built an LSTM model using TensorFlow/Keras.
     - Configured layers: LSTM, Dense, Dropout.
  3. **Training:**
     - Split data into training and validation sets.
     - Trained the model using Mean Squared Error (MSE) as the loss function.
  4. **Prediction:**
     - Used the trained model to predict future closing prices.
     - Compared predictions with actual values to evaluate performance.

---

## Tools and Libraries
- **Data Collection:** `yfinance`
- **Data Analysis and Visualization:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
- **Risk Analysis:** `Monte Carlo Simulation`
- **Prediction:** `TensorFlow`, `Keras`, `Scikit-learn`

---

## Results
- **Analysis Insights:**
  - Identified trends, daily returns, and correlations in stock prices.
  - Assessed the risk associated with different stocks.
- **Prediction Outcome:**
  - Predicted closing prices of Apple Inc. with reasonable accuracy using the LSTM model.
  - Highlighted the potential of LSTM for time-series forecasting in finance.

---

## How to Run the Project
1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd stock-market-analysis
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook:**
   - Open `stock_analysis.ipynb` in Jupyter Notebook or Google Colab.
   - Execute the cells sequentially to perform analysis and prediction.

4. **View Results:**
   - Check the generated plots and prediction outputs.
   - Evaluate the LSTM model's performance metrics.

---

## Summary
In this project, we:
- Explored and visualized stock market data.
- Measured financial metrics such as daily returns, moving averages, and correlations.
- Assessed the risk of investing in stocks.
- Built an LSTM model to predict Apple Inc.'s stock price.

This comprehensive approach provides a solid foundation for understanding stock market behavior and leveraging machine learning for financial forecasting.

