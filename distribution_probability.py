import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Force the Streamlit page layout to use the full screen width
st.set_page_config(layout="wide")

# App Title & Subtitle
st.title("Stock Return Probability Distribution")
st.write("Enter up to 4 stock tickers and press **Enter** to visualize their annual return distributions and Sharpe Ratios.")

# User Input
tickers_input = st.text_input("Enter Tickers (separated by commas)", "F, AAPL, MSFT, GOOG")
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

if len(tickers) > 4:
    st.error("Please enter a maximum of 4 tickers.")
    st.stop()

# Cache function to fetch data for a single stock ticker
@st.cache_data
def load_single_ticker_data(ticker):
    data = yf.download(ticker, period="5y", progress=False)
    return data

# Dynamic risk-free rate fetching function using the 13-week T-Bill yield (^IRX)
@st.cache_data
def get_risk_free_rate():
    try:
        # Fetch the most recent 13-week Treasury Bill ticker yield
        t_bill = yf.download("^IRX", period="5d", progress=False)
        if not t_bill.empty:
            # ^IRX is returned as a percentage (e.g., 4.5 means 4.5%). Divide by 100 to get decimal form.
            latest_rate = t_bill['Close'].squeeze().iloc[-1] / 100
            return latest_rate
    except Exception:
        pass
    # Safe default fallback (e.g., historical average 4.0%) if fetching fails
    return 0.040

if tickers:
    plots = []
    
    with st.spinner("Fetching data and analyzing distributions..."):
        # Acquire real-time benchmark risk-free rate
        rf_rate = get_risk_free_rate()
        
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
                std_dev = annual_returns.std() # Volatility
                minus_1_sd = mean_return - std_dev
                plus_1_sd = mean_return + std_dev
                
                # Calculate Sharpe Ratio: (Mean Return - Risk Free Rate) / Standard Deviation
                # Prevent DivisionByZero errors if standard deviation is somehow 0
                if std_dev > 0:
                    sharpe_ratio = (mean_return - rf_rate) / std_dev
                    sharpe_text = f" | Sharpe Ratio: {sharpe_ratio:.2f}"
                else:
                    sharpe_text = " | Sharpe Ratio: N/A"
                
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

                # Filter curve data points that fall strictly between -1 SD and +1 SD for inner shading
                sd_zone_df = curve_df[(curve_df['Annual Return'] >= minus_1_sd) & (curve_df['Annual Return'] <= plus_1_sd)]

                # Build chart blank canvas
                fig = go.Figure()

                # 1. Outer Shaded Area (Entire distribution profile - ultra soft translucent light blue)
                fig.add_trace(
                    go.Scatter(
                        x=curve_df['Annual Return'],
                        y=curve_df['Probability Density'],
                        fill='tozeroy',
                        fillcolor="rgba(173, 216, 230, 0.3)", 
                        mode='none',                         
                        line=dict(shape='spline'),           
                        hoverinfo='skip',
                        showlegend=False
                    )
                )

                # 2. Inner Shaded Area (Core Standard Deviation range - deeper translucent blue)
                fig.add_trace(
                    go.Scatter(
                        x=sd_zone_df['Annual Return'],
                        y=sd_zone_df['Probability Density'],
                        fill='tozeroy',
                        fillcolor="rgba(74, 144, 226, 0.45)", 
                        mode='none',                          
                        line=dict(shape='spline'),
                        hoverinfo='skip',
                        showlegend=False
                    )
                )
                
                # 3. Add vertical drop lines down to the X-axis with clean top text markers
                # Mean Line (Black Dotted)
                fig.add_vline(
                    x=mean_return, 
                    line_dash="dot", 
                    line_color="black", 
                    line_width=1.5,
                    annotation_text=f"Mean: {mean_return:.2%}", 
                    annotation_position="top right",
                    annotation_font=dict(color="black", size=11)
                )
                # -1 SD Line (Gray Dotted)
                fig.add_vline(
                    x=minus_1_sd, 
                    line_dash="dot", 
                    line_color="gray", 
                    line_width=1,
                    annotation_text=f"-1 SD: {minus_1_sd:.2%}", 
                    annotation_position="top left",
                    annotation_font=dict(color="gray", size=10)
                )
                # +1 SD Line (Gray Dotted)
                fig.add_vline(
                    x=plus_1_sd, 
                    line_dash="dot", 
                    line_color="gray", 
                    line_width=1,
                    annotation_text=f"+1 SD: {plus_1_sd:.2%}", 
                    annotation_position="top right",
                    annotation_font=dict(color="gray", size=10)
                )
                
                # Apply layout, axis, and include the Sharpe Ratio inside the Title block
                fig.update_layout(
                    title=f"{ticker} Annual Return Distribution (5yr){sharpe_text}",
                    xaxis_title="Annual Return (Log Scale)",
                    yaxis_title="Density",
                    showlegend=False,
                    plot_bgcolor="#F4F4F6",  
                    paper_bgcolor="#FFFFFF", 
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(gridcolor="#FFFFFF"), 
                    yaxis=dict(gridcolor="#FFFFFF")
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
st.sidebar.info(f"This application tracks historical metrics. Current benchmark risk-free rate used for Sharpe calculation: {rf_rate:.2%}")
