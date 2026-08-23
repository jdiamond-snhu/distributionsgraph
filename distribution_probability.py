import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# Force the Streamlit page layout to use the full screen width
st.set_page_config(layout="wide")

# App Title & Subtitle (Flush left automatically)
st.title("Stock Return Probability Distribution")
st.write("Enter up to 4 stock tickers and press **Enter** to visualize their annual return distributions.")

# User Input
tickers_input = st.text_input("Enter Tickers (separated by commas)", "F, AAPL, MSFT, GOOG")
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

if len(tickers) > 4:
    st.error("Please enter a maximum of 4 tickers.")
    st.stop()

# Isolated data fetch function for a single ticker to prevent multi-index data clash
@st.cache_data
def load_single_ticker_data(ticker):
    data = yf.download(ticker, period="5y", progress=False)
    return data

if tickers:
    plots = []
    
    with st.spinner("Fetching data and analyzing distributions..."):
        for ticker in tickers:
            try:
                # Safely pull isolated DataFrame for the specific stock
                df = load_single_ticker_data(ticker)
                
                # Check if DataFrame has rows natively
                if df is None or len(df) == 0:
                    st.warning(f"No data found for {ticker}")
                    continue
                
                # Check for either 'Adj Close' or 'Close' column safely
                price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                
                # Calculate daily log returns using the single column array
                # .squeeze() ensures data handles as a flat series if yfinance pads it
                prices = df[price_col].squeeze()
                
                log_returns = np.log(prices / prices.shift(1))
                
                # Annualization scaling factor (approx. 252 trading days)
                annual_returns = log_returns.dropna() * 252
                
                if annual_returns.empty:
                    st.warning(f"Not enough return data to plot {ticker}")
                    continue

                # Calculate the exact mean return
                mean_return = annual_returns.mean()
                
                # Create a clean DataFrame for Plotly Express
                plot_df = pd.DataFrame({
                    'Annual Return': annual_returns
                })

                # Native Plotly histogram (No scipy dependency)
                fig = px.histogram(
                    plot_df, 
                    x="Annual Return",
                    nbins=40,
                    title=f"{ticker} Annual Return Distribution (5yr)",
                    labels={"Annual Return": "Annual Return (Log Scale)"},
                    template="plotly_white",
                    opacity=0.7
                )
                
                # Highlight the calculated mean return
                fig.add_vline(
                    x=mean_return, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Mean: {mean_return:.2%}", 
                    annotation_position="top right"
                )
                
                fig.update_layout(
                    yaxis_title="Frequency Count",
                    showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20) # Tighten margins for grid alignment
                )
                
                plots.append(fig)
                
            except Exception as e:
                st.error(f"Error processing {ticker}: {e}")

    # Render plots in a 2x2 grid structure
    if plots:
        # Create a loop that processes 2 elements at a time
        for i in range(0, len(plots), 2):
            # Create two equal-width columns for this row
            col1, col2 = st.columns(2)
            
            # Place the first plot in the left column
            with col1:
                st.plotly_chart(plots[i], use_container_width=True)
                
            # Place the second plot in the right column if it exists
            if i + 1 < len(plots):
                with col2:
                    st.plotly_chart(plots[i+1], use_container_width=True)

st.sidebar.header("About")
st.sidebar.info("This application tracks historical metrics and shifts directly on text submission.")
