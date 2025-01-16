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

# Function to fetch stock data
def get_stock_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    if stock_data.empty:
        raise ValueError(f"No data found for tickers: {tickers}")

    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data = stock_data['Close']

    if stock_data.empty:
        raise KeyError("'Close' column not found in the downloaded data.")

    returns = stock_data.pct_change().dropna()
    return returns


# Function to fetch Fama-French factors
def get_fama_french_factors():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch data from Ken French's site.")
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        file_name = [name for name in z.namelist() if name.endswith('.CSV')][0]
        with z.open(file_name) as csvfile:
            factors = pd.read_csv(csvfile, skiprows=4)
    
    factors.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
    factors = factors[factors['Date'].apply(lambda x: str(x).isdigit())]
    factors['Date'] = pd.to_datetime(factors['Date'], format='%Y%m%d')
    factors.set_index('Date', inplace=True)
    factors = factors.astype(float) / 100  # Convert percentages to decimals
    return factors


# Function to calculate optimal portfolio weights
def calculate_weights(returns, fama_french_factors):
    def objective(weights):
        portfolio_returns = np.dot(returns.values, weights)
        excess_returns = portfolio_returns - fama_french_factors['RF'].values
        X = fama_french_factors[['Mkt-RF', 'SMB', 'HML']].values
        beta, _, _, _ = np.linalg.lstsq(X, excess_returns, rcond=None)
        residuals = excess_returns - np.dot(X, beta)
        return np.var(residuals)
    
    n = len(returns.columns)
    initial_weights = np.ones(n) / n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
    return result.x


# Function to plot contribution charts
def plot_contribution_chart(weights, returns, fama_french_factors):
    portfolio_returns = np.dot(returns.values, weights)
    excess_returns = portfolio_returns - fama_french_factors['RF'].values
    X = fama_french_factors[['Mkt-RF', 'SMB', 'HML']].values
    beta, _, _, _ = np.linalg.lstsq(X, excess_returns, rcond=None)
    contributions = beta * weights[:, None]

    factors = ['Mkt-RF', 'SMB', 'HML']
    for i, factor in enumerate(factors):
        plt.figure()
        plt.bar(returns.columns, contributions[:, i], alpha=0.7)
        plt.title(f'Contributions to {factor}')
        plt.xlabel('Stock')
        plt.ylabel('Contribution')
        st.pyplot(plt)


# Function to compute beta matrix
def compute_beta_matrix(returns, fama_french_factors):
    X = fama_french_factors[['Mkt-RF', 'SMB', 'HML']].values
    beta_matrix = []
    for stock in returns.columns:
        y = returns[stock] - fama_french_factors['RF'].values
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        beta_matrix.append(beta)
    return pd.DataFrame(beta_matrix, columns=['Mkt-RF', 'SMB', 'HML'], index=returns.columns)


# Streamlit Interface
st.title("Fama-French Portfolio Optimization")
st.sidebar.header("Input Parameters")

# User inputs
tickers = st.sidebar.text_input("Enter stock tickers (comma-separated):", "AAPL, MSFT, TSLA")
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.today())

if st.sidebar.button("Run Analysis"):
    tickers = [ticker.strip() for ticker in tickers.split(",")]
    stock_returns = get_stock_data(tickers, start_date, end_date)
    fama_french_factors = get_fama_french_factors()

    # Align data
    fama_french_factors = fama_french_factors.loc[stock_returns.index.min():stock_returns.index.max()]
    weights = calculate_weights(stock_returns, fama_french_factors)

    # Display results
    st.subheader("Optimal Portfolio Weights")
    portfolio_df = pd.DataFrame({'Stock': stock_returns.columns, 'Weight': weights})
    st.dataframe(portfolio_df)

    st.subheader("Sharpe Ratio")
    portfolio_returns = np.dot(stock_returns.values, weights)
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std()
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Contribution charts
    st.subheader("Contribution Charts")
    plot_contribution_chart(weights, stock_returns, fama_french_factors)

    # Factor sensitivity heatmap
    st.subheader("Factor Sensitivity Heatmap")
    beta_matrix = compute_beta_matrix(stock_returns, fama_french_factors)
    st.dataframe(beta_matrix)
    plt.figure(figsize=(10, 6))
    sns.heatmap(beta_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(stock_returns.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)
