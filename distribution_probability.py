import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.figure_factory as ff

# App Title
st.title("Stock Return Probability Distribution")
st.write("Enter up to 4 stock tickers to visualize their annual return distributions.")

# User Input
tickers_input = st.text_input("Enter Tickers (separated by commas)", "AAPL, MSFT, GOOG, AMZN")
tickers = [t.strip().upper() for t in tickers_input.split(',')]

if len(tickers) > 4:
    st.error("Please enter a maximum of 4 tickers.")
    st.stop()

# Data Fetching
@st.cache_data
def load_data(ticker):
    # Fetch 5 years of data
    data = yf.download(ticker, period="5y")
    return data

if st.button("Generate Graphs"):
    
    plots = []
    
    for ticker in tickers:
        try:
            with st.spinner(f"Analyzing {ticker}..."):
                data = load_data(ticker)
                
                if data.empty:
                    st.warning(f"No data found for {ticker}")
                    continue
                
                # Calculate Daily Returns
                data['Daily Return'] = data['Adj Close'].pct_change()
                
                # Calculate Annualized Returns (approx. 252 trading days)
                # Using log returns for better additive properties over time
                data['Log Return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
                
                # Simple approximation for annual return distribution
                annual_returns = data['Log Return'].dropna() * 252
                
                # Plotly Distribution Plot
                fig = ff.create_distplot(
                    [annual_returns], 
                    [ticker], 
                    bin_size=0.05, 
                    show_rug=False
                )
                
                fig.update_layout(
                    title=f"{ticker} Annual Return Distribution (5yr)",
                    xaxis_title="Annual Return (Log)",
                    yaxis_title="Density",
                    template="plotly_white"
                )
                
                plots.append(fig)
        
        except Exception as e:
            st.error(f"Error processing {ticker}: {e}")

    # Display Plots in Streamlit
    for plot in plots:
        st.plotly_chart(plot, use_container_width=True)

st.sidebar.header("About")
st.sidebar.info("This app calculates the probability distribution of annual returns based on 5 years of historical data using yfinance and plotly.")
