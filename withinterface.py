import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize
import zipfile
import requests
import io
from scipy.stats import norm

# Functions for Project

# Function to upload stock data from yfinance with possible raising error
def get_stock_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    if stock_data.empty:
        raise ValueError(f"Data were not found for these tickers: {tickers}")

    if isinstance(stock_data.columns, pd.MultiIndex): #check if DataFrame is MultiIndex as code handle DataFrame with hierarchical column
        stock_data = stock_data['Close']

    if stock_data.empty:
        raise KeyError("'Close' column not found in the downloaded data.")

    returns = stock_data.pct_change().dropna()
    return returns

# Function upload Fama-French factors from Dartmouth database. It was done with the help of ChatGPT to make sure data was retrieved successfully.
def get_fama_french_factors():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    response = requests.get(url)
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        file_name = [name for name in z.namelist() if name.endswith('.CSV')][0]
        with z.open(file_name) as csvfile:
            factors = pd.read_csv(csvfile, skiprows=4)
            
    #To apply marks for factors and convert percentages to decimals
    factors.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
    factors = factors[factors['Date'].apply(lambda x: str(x).isdigit())]
    factors['Date'] = pd.to_datetime(factors['Date'], format='%Y%m%d')
    factors.set_index('Date', inplace=True)
    factors = factors.astype(float) / 100  # Convert percentages to decimals
    return factors

# Function to estimate optimal portfolio weights
def calculate_weights(returns, fama_french_factors, selected_factors):
    def objective(weights):
        portfolio_returns = np.dot(returns.values, weights)
        excess_returns = portfolio_returns - fama_french_factors['RF'].values
        X = fama_french_factors[selected_factors].values
        beta, _, _, _ = np.linalg.lstsq(X, excess_returns, rcond=None) #performs linear regression to estimate beta
        residuals = excess_returns - np.dot(X, beta)
        return np.var(residuals)

    # Optimization part starting with equal weights and constraint(sum of all weights should 1) and every weight should be from 0 to 1
    # Optimization was done using scipy minimization.
    n = len(returns.columns)
    initial_weights = np.ones(n) / n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
    return result.x

# Function to calculate regression statistics
def regression_statistics(X, y):
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    y_predicted = np.dot(X, beta)
    n, k = X.shape
    rsum_of_squares = np.sum((y - y_predicted)**2)
    total_sum_of_squares = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (rsum_of_squares / total_sum_of_squares)

    mean_squared_error = rsum_of_squares / (n - k)
    std_error = np.sqrt(np.diagonal(mean_squared_error * np.linalg.inv(np.dot(X.T, X))))
    t_stat = beta / std_error
    p_value = 2 * (1 - norm.cdf(np.abs(t_stat)))

    return beta, std_error, t_stat, p_value, r_squared

# Function to plot contribution charts. Contributions are calculated by multiplying beta with the weight
def plot_contribution_chart(weights, returns, fama_french_factors, selected_factors):
    portfolio_returns = np.dot(returns.values, weights)
    excess_returns = portfolio_returns - fama_french_factors['RF'].values
    X = fama_french_factors[selected_factors].values
    beta, _, _, _ = np.linalg.lstsq(X, excess_returns, rcond=None)
    contributions = beta * weights[:, None]

    for i, factor in enumerate(selected_factors):
        plt.figure()
        plt.bar(returns.columns, contributions[:, i], alpha=0.8)
        plt.title(f'Contributions to {factor}')
        plt.xlabel('Stock')
        plt.ylabel('Contribution')
        st.pyplot(plt)

# Function to compute beta matrix
def compute_beta_matrix(returns, fama_french_factors, selected_factors):
    X = fama_french_factors[selected_factors].values
    beta_matrix = []
    for stock in returns.columns:
        y = returns[stock] - fama_french_factors['RF'].values
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        beta_matrix.append(beta)
    return pd.DataFrame(beta_matrix, columns=selected_factors, index=returns.columns)


# Streamlit Interface
st.title("Fama-French Portfolio Optimization")
st.sidebar.header("Input Parameters")

# User inputs
tickers = st.sidebar.text_input("Enter stock tickers (comma-separated):", "AAPL,SYM,BMRA,UNH,BBAI,NUKK")
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2024-12-02"))

# Factor selection. This part was done with some help from the ChatGPT
st.sidebar.subheader("Select Factors to Use")
use_all_factors = st.sidebar.checkbox("Use All Factors", value=True)
selected_factors = []

if not use_all_factors:
    if st.sidebar.checkbox("Market (Mkt-RF)"):
        selected_factors.append("Mkt-RF")
    if st.sidebar.checkbox("Size (SMB)"):
        selected_factors.append("SMB")
    if st.sidebar.checkbox("Value (HML)"):
        selected_factors.append("HML")
else:
    selected_factors = ["Mkt-RF", "SMB", "HML"]

# Data uploading
if st.sidebar.button("Run Analysis"):
    tickers = [ticker.strip() for ticker in tickers.split(",")]
    stock_returns = get_stock_data(tickers, start_date, end_date)
    fama_french_factors = get_fama_french_factors()

    # Align data
    fama_french_factors = fama_french_factors.loc[stock_returns.index.min():stock_returns.index.max()]
    weights = calculate_weights(stock_returns, fama_french_factors, selected_factors)

    # For results
    st.subheader("Optimal Portfolio Weights")
    portfolio_df = pd.DataFrame({'Stock': stock_returns.columns, 'Weight': weights})
    st.dataframe(portfolio_df)

    # Returns presenting
    st.subheader("Portfolio Returns")
    portfolio_returns = np.dot(stock_returns.values, weights)
    st.line_chart(pd.DataFrame(portfolio_returns, index=stock_returns.index, columns=['Portfolio Returns']))
   
    # Calculate and display volatility
    st.subheader("Volatility")
    stock_volatility = stock_returns.std() * np.sqrt(252)  # Annualized volatility
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(stock_returns.cov() * 252, weights)))  # Annualized portfolio volatility, 252 is trading period
    st.write("Individual Stock Volatility (Annualized):")
    st.dataframe(stock_volatility.rename("Volatility"))
    st.write(f"Portfolio Volatility (Annualized): {portfolio_volatility:.2%}")

    # Calculation of Sharpe Ratio
    st.subheader("Sharpe Ratio")
    portfolio_returns = np.dot(stock_returns.values, weights)
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std()
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Statistical Data
    st.subheader("Regression Statistics")
    X = fama_french_factors[selected_factors].values
    y = portfolio_returns - fama_french_factors['RF'].values
    beta, std_error, t_stat, p_value, r_squared = regression_statistics(X, y)
    regression_results = pd.DataFrame({
        'Factor': selected_factors,
        'Beta': beta,
        'Std. Error': std_error,
        't-Statistic': t_stat,
        'p-Value': p_value
    })
    st.dataframe(regression_results)
    st.write(f"R-squared: {r_squared:.4f}")    
    
    # Contribution charts
    st.subheader("Contribution Charts")
    plot_contribution_chart(weights, stock_returns, fama_french_factors, selected_factors)

    # Factor sensitivity heatmap
    st.subheader("Factor Sensitivity Heatmap")
    beta_matrix = compute_beta_matrix(stock_returns, fama_french_factors, selected_factors)
    st.dataframe(beta_matrix)
    plt.figure(figsize=(10, 6))
    sns.heatmap(beta_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(stock_returns.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)


