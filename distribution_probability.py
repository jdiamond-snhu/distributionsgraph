import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Force the Streamlit page layout to use the full screen width
st.set_page_config(layout="wide")

# App Title & Subtitle
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
                prices = df[price_col].squeeze()
                log_returns = np.log(prices / prices.shift(1))
                
                # Annualization scaling factor (approx. 252 trading days)
                annual_returns = log_returns.dropna() * 252
                
                if annual_returns.empty:
                    st.warning(f"Not enough return data to plot {ticker}")
                    continue

                # Calculate statistical parameters
                mean_return = annual_returns.mean()
                std_dev = annual_returns.std()
                minus_1_sd = mean_return - std_dev
                plus_1_sd = mean_return + std_dev
                
                # Generate smooth frequency curve data natively without scipy
                counts, bin_edges = np.histogram(annual_returns, bins=50, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Create a DataFrame of the distribution path
                curve_df = pd.DataFrame({
                    'Annual Return': bin_centers,
                    'Probability Density': counts
                })
                
                # Smooth the data using a rolling average for an elegant curve shape
                curve_df['Probability Density'] = curve_df['Probability Density'].rolling(window=3, center=True, min_periods=1).mean()

                # Generate the base area chart
                fig = px.area(
                    curve_df, 
                    x="Annual Return",
                    y="Probability Density",
                    title=f"{ticker} Annual Return Distribution (5yr)",
                    labels={"Annual Return": "Annual Return (Log Scale)", "Probability Density": "Density"},
                    template="plotly_white",
                    color_discrete_sequence=["#4A90E2"] 
                )
                
                # Make the line smooth (spline) and fill it with translucent light blue
                fig.update_traces(
                    line_shape="spline",
                    line_width=2.5,
                    fill='tozeroy',
                    fillcolor="rgba(173, 216, 230, 0.4)" 
                )
                
                # Helper function to find the closest Y (Density) coordinate on the curve for a given X value
                def get_curve_y(x_val):
                    idx = (curve_df['Annual Return'] - x_val).abs().idxmin()
                    return curve_df.loc[idx, 'Probability Density']

                # Find exact plot intersection coordinates
                mean_y = get_curve_y(mean_return)
                minus_sd_y = get_curve_y(minus_1_sd)
                plus_sd_y = get_curve_y(plus_1_sd)

                # Add a scatter trace layer for the 3 visual highlight dots
                fig.add_trace(
                    go.Scatter(
                        x=[minus_1_sd, mean_return, plus_1_sd],
                        y=[minus_sd_y, mean_y, plus_sd_y],
                        mode="markers+text",
                        marker=dict(color="#1F77B4", size=10, symbol="circle"), # Distinct blue dots
                        text=[f"-1 SD: {minus_1_sd:.2%}", f"Mean: {mean_return:.2%}", f"+1 SD: {plus_1_sd:.2%}"],
                        textposition=["top left", "top center", "top right"],
                        textfont=dict(color="black", size=11),
                        hoverinfo="skip"
                    )
                )
                
                # Draw vertical lines down to the X-axis
                # Mean Line (Black Dotted)
                fig.add_vline(x=mean_return, line_dash="dot", line_color="black", line_width=1.5)
                # -1 SD Line (Gray Dotted)
                fig.add_vline(x=minus_1_sd, line_dash="dot", line_color="gray", line_width=1)
                # +1 SD Line (Gray Dotted)
                fig.add_vline(x=plus_1_sd, line_dash="dot", line_color="gray", line_width=1)
                
                fig.update_layout(
                    showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                plots.append(fig)
                
            except Exception as e:
                st.error(f"Error processing {ticker}: {e}")

    # Render plots in a 2x2 grid structure
    if plots:
        for i in range(0, len(plots), 2):
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(plots[i], use_container_width=True)
                
            if i + 1 < len(plots):
                with col2:
                    st.plotly_chart(plots[i+1], use_container_width=True)

st.sidebar.header("About")
st.sidebar.info("This application tracks historical metrics and shifts directly on text submission.")
