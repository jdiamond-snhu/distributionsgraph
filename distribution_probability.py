import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.figure_factory as ff

# App Title
st.title("Stock Return Probability Distribution")
st.write("Enter up to 4 stock tickers and press **Enter** to visualize their annual return distributions.")

# User Input - Pressing Enter automatically re-runs the script
tickers_input = st.text_input("Enter Tickers (separated by commas)", "F, AAPL, MSFT")
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

if len(tickers) > 4:
    st.error("Please enter a maximum of 4 tickers.")
    st.stop()

# Data Fetching
@st.cache_data
def load_data(ticker_list):
    # Fetch 5 years of data for all tickers at once to keep multi-indexing uniform
    # group_by='ticker' ensures clean column access
    data = yf.download(ticker_list, period="5y", group_by='ticker')
    return data

if tickers:
    plots = []
    
    with st.spinner("Fetching data and analyzing distributions..."):
        raw_data = load_data(tickers)
        
        for ticker in tickers:
            try:
                # Handle single vs multiple ticker structures from yfinance
                if len(tickers) == 1:
                    df = raw_data
                else:
                    if ticker not in raw_data.columns.levels[0]:
                        st.warning(f"No data found for {ticker}")
                        continue
                    df = raw_data[ticker]
                
                if df.empty:
                    st.warning(f"No data found for {ticker}")
                    continue
                
                # Check for either 'Adj Close' or 'Close' column safely
                price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                
                # Calculate daily log returns
                df['Log Return'] = np.log(df[price_col] / df[price_col].shift(1))
                
                # Broad annualization scaling factor (approx. 252 trading days)
                annual_returns = df['Log Return'].dropna() * 252
                
                if annual_returns.empty:
                    st.warning(f"Not enough return data to plot {ticker}")
                    continue

                # Create interactive Plotly distribution curve
                fig = ff.create_distplot(
                    [annual_returns], 
                    [ticker], 
                    bin_size=0.05, 
                    show_rug=False
                )
                
                fig.update_layout(
                    title=f"{ticker} Annual Return Distribution (5yr Curve)",
                    xaxis_title="Annual Return (Log Scale)",
                    yaxis_title="Probability Density",
                    template="plotly_white"
                )
                
                plots.append(fig)
                
            except Exception as e:
                st.error(f"Error processing {ticker}: {e}")

    # Render all successfully built plots
    for plot in plots:
        st.plotly_chart(plot, use_container_width=True)

st.sidebar.header("About")
st.sidebar.info("This application tracks historical metrics and shifts directly on text submission.")
