import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# App Title
st.title("Stock Return Probability Distribution")
st.write("Enter up to 4 stock tickers and press **Enter** to visualize their annual return distributions.")

# User Input
tickers_input = st.text_input("Enter Tickers (separated by commas)", "F, AAPL, MSFT")
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

if len(tickers) > 4:
    st.error("Please enter a maximum of 4 tickers.")
    st.stop()

# Data Fetching
@st.cache_data
def load_data(ticker_list):
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
                    if ticker not in raw_data.columns.levels:
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
                
                # Annualization scaling factor (approx. 252 trading days)
                annual_returns = df['Log Return'].dropna() * 252
                
                if annual_returns.empty:
                    st.warning(f"Not enough return data to plot {ticker}")
                    continue

                # Calculate the exact mean for display
                mean_return = annual_returns.mean()
                
                # Create a clean DataFrame for Plotly Express
                plot_df = pd.DataFrame({
                    'Annual Return': annual_returns
                })

                # Native Plotly histogram (Does NOT require scipy)
                fig = px.histogram(
                    plot_df, 
                    x="Annual Return",
                    nbins=40,
                    title=f"{ticker} Annual Return Distribution (5yr)",
                    labels={"Annual Return": "Annual Return (Log Scale)"},
                    template="plotly_white",
                    opacity=0.7
                )
                
                # Add a vertical line exactly at the mean return
                fig.add_vline(
                    x=mean_return, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Mean: {mean_return:.2%}", 
                    annotation_position="top right"
                )
                
                fig.update_layout(
                    yaxis_title="Frequency Count",
                    showlegend=False
                )
                
                plots.append(fig)
                
            except Exception as e:
                st.error(f"Error processing {ticker}: {e}")

    # Render all charts
    for plot in plots:
        st.plotly_chart(plot, use_container_width=True)

st.sidebar.header("About")
st.sidebar.info("This application tracks historical metrics and shifts directly on text submission.")
