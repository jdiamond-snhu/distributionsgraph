import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Force the Streamlit page layout to use the full screen width
st.set_page_config(layout="wide")

# App Title & Subtitle
st.title("Stock Return Probability Distribution")
st.write("Enter up to 4 stock tickers and press **Enter** to visualize their expected annual return distributions.")

# User Input
tickers_input = st.text_input("Enter Tickers (separated by commas)", "MCD, AAPL, MSFT, GOOG")
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
        t_bill = yf.download("^IRX", period="5d", progress=False)
        if not t_bill.empty:
            latest_rate = t_bill['Close'].squeeze().iloc[-1] / 100
            return latest_rate
    except Exception:
        pass
    return 0.040 # Safe 4% fallback if fetching fails

if tickers:
    plots = []
    
    with st.spinner("Fetching data and analyzing distributions..."):
        rf_rate = get_risk_free_rate()
        
        for ticker in tickers:
            try:
                # Safely pull isolated DataFrame for the specific stock
                df = load_single_ticker_data(ticker)
                
                if df is None or len(df) == 0:
                    st.warning(f"No data found for {ticker}")
                    continue
                
                price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                prices = df[price_col].squeeze()
                
                # Calculate daily log returns
                log_returns = np.log(prices / prices.shift(1)).dropna()
                
                if log_returns.empty:
                    st.warning(f"Not enough return data to plot {ticker}")
                    continue

                # --- CORRECT MATHEMATICAL ANNUALIZATION SCALE ---
                daily_mean = log_returns.mean()
                daily_std = log_returns.std()
                
                mean_return = daily_mean * 252                # Mean scales linearly
                std_dev = daily_std * np.sqrt(252)            # Volatility scales by square root of time
                
                minus_1_sd = mean_return - std_dev
                plus_1_sd = mean_return + std_dev
                
                # Calculate True Sharpe Ratio
                if std_dev > 0:
                    sharpe_ratio = (mean_return - rf_rate) / std_dev
                    sharpe_text = f" | Sharpe Ratio: {sharpe_ratio:.2f}"
                else:
                    sharpe_text = " | Sharpe Ratio: N/A"
                
                # Generate a clean parametric normal probability curve (No scipy dependency)
                # Generate 200 clean coordinate points between -3.5 and +3.5 Standard Deviations
                x_values = np.linspace(mean_return - 3.5 * std_dev, mean_return + 3.5 * std_dev, 200)
                y_values = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean_return) / std_dev) ** 2)
                
                curve_df = pd.DataFrame({
                    'Annual Return': x_values,
                    'Probability Density': y_values
                })

                # Filter curve data points that fall strictly between -1 SD and +1 SD for the core shading
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
                    annotation_text=f"Mean: {mean_return:.1%}", 
                    annotation_position="top right",
                    annotation_font=dict(color="black", size=11)
                )
                # -1 SD Line (Gray Dotted)
                fig.add_vline(
                    x=minus_1_sd, 
                    line_dash="dot", 
                    line_color="gray", 
                    line_width=1,
                    annotation_text=f"-1 SD: {minus_1_sd:.1%}", 
                    annotation_position="top left",
                    annotation_font=dict(color="gray", size=10)
                )
                # +1 SD Line (Gray Dotted)
                fig.add_vline(
                    x=plus_1_sd, 
                    line_dash="dot", 
                    line_color="gray", 
                    line_width=1,
                    annotation_text=f"+1 SD: {plus_1_sd:.1%}", 
                    annotation_position="top right",
                    annotation_font=dict(color="gray", size=10)
                )
                
                # Apply layout, format the X-axis as percentage scales
                fig.update_layout(
                    title=f"{ticker} Expected Annual Return Profile (5yr Data){sharpe_text}",
                    xaxis_title="Expected Annual Return (%)",
                    yaxis_title="Probability Density",
                    showlegend=False,
                    plot_bgcolor="#F4F4F6",  
                    paper_bgcolor="#FFFFFF", 
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(
                        gridcolor="#FFFFFF",
                        tickformat=".1%" # Displays numbers clean like -10.0%, 0.0%, 10.0%
                    ), 
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
st.sidebar.info(f"Current risk-free benchmark yield used for Sharpe calculation: {rf_rate:.2%}")
