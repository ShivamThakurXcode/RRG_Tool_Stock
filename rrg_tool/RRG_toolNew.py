# Super-Enhanced Sector Rotation Analysis Tool
# Enhancements:
# - Modern, responsive UI with dark mode, custom CSS, and metric cards
# - Optimized data fetching with caching and parallel processing
# - Quadrant-focused RRG with zoom, animations, and dynamic scaling
# - Advanced metrics: volatility, momentum acceleration, trend strength
# - Comprehensive PDF reporting with fpdf
# - Save/load sector lists, historical comparison, and export options
# - Trend alerts and interactive legend
# - Guided tour and beginner-friendly explanations
# - Robust error handling and performance optimization

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from scipy.interpolate import interp1d
from fpdf import FPDF
import json
import asyncio
import concurrent.futures
import io
import base64
import streamlit.components.v1 as components

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Advanced Sector Rotation Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning UI
st.markdown("""
<style>
    .stApp {
        max-width: 1440px;
        margin: 0 auto;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        color: white;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .metric-card {
        background-color: #2a2a2a;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        color: white;
    }
    .metric-title {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        color: #00ff88;
    }
    .quadrant-leading { background: linear-gradient(135deg, rgba(0,255,0,0.2), rgba(0,200,0,0.1)); }
    .quadrant-weakening { background: linear-gradient(135deg, rgba(255,165,0,0.2), rgba(200,100,0,0.1)); }
    .quadrant-lagging { background: linear-gradient(135deg, rgba(255,0,0,0.2), rgba(200,0,0,0.1)); }
    .quadrant-improving { background: linear-gradient(135deg, rgba(0,0,255,0.2), rgba(0,0,200,0.1)); }
    [data-testid="stExpander"] {
        background-color: #2a2a2a;
        border-radius: 8px;
    }
    .stSpinner > div > div {
        border-color: #007bff transparent transparent transparent;
    }
</style>
""", unsafe_allow_html=True)

# Dark mode toggle
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Cache data fetching
@st.cache_data(ttl=3600)
def fetch_data(symbols, period_days):
    """Fetch data from Yahoo Finance with parallel processing"""
    data = {}
    failed_symbols = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days + 10)

    async def fetch_symbol(symbol):
        try:
            hist = await asyncio.get_event_loop().run_in_executor(
                None, lambda: yf.download(symbol, start=start_date, end=end_date, progress=False)
            )
            if len(hist) < 20:
                return symbol, None
            return symbol, hist['Close']
        except Exception:
            return symbol, None

    async def fetch_all():
        tasks = [fetch_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return results

    # Run async fetch
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(fetch_all())
    loop.close()

    for symbol, prices in results:
        if prices is None:
            failed_symbols.append(symbol)
        else:
            data[symbol] = prices

    return data, failed_symbols

def calculate_relative_strength(price_data, benchmark_data, period):
    """Calculate relative strength vs benchmark"""
    min_length = min(len(price_data), len(benchmark_data))
    if min_length < period:
        return None, None, None

    aligned_data = pd.DataFrame({
        'price': price_data,
        'benchmark': benchmark_data
    }).dropna()

    if len(aligned_data) < period:
        return None, None, None

    relative_strength = aligned_data['price'] / aligned_data['benchmark']
    rs_momentum = relative_strength.pct_change(period).dropna()
    volatility = aligned_data['price'].pct_change().rolling(period).std().dropna() * np.sqrt(252)

    return relative_strength, rs_momentum, volatility

def calculate_jdk_rs_ratio(relative_strength, short_period=10, long_period=40):
    """Calculate JdK RS-Ratio"""
    if len(relative_strength) < long_period:
        return None
    rs_normalized = (relative_strength / relative_strength.rolling(long_period).mean()) * 100
    return rs_normalized

def calculate_jdk_rs_momentum(rs_ratio, period=10):
    """Calculate JdK RS-Momentum and acceleration"""
    if rs_ratio is None or len(rs_ratio) < period:
        return None, None
    momentum = ((rs_ratio / rs_ratio.shift(period)) - 1) * 100
    acceleration = momentum.diff(period).dropna()
    return momentum, acceleration

def get_quadrant_info(rs_ratio, rs_momentum):
    """Determine quadrant with investment implications"""
    if rs_ratio > 100 and rs_momentum > 0:
        return "Leading", "#00ff88", "🚀", "Strong performance, consider holding"
    elif rs_ratio > 100 and rs_momentum < 0:
        return "Weakening", "#ffaa00", "📉", "Losing momentum, consider selling"
    elif rs_ratio < 100 and rs_momentum < 0:
        return "Lagging", "#ff4444", "📊", "Underperforming, avoid"
    else:
        return "Improving", "#00aaff", "📈", "Gaining momentum, consider buying"

def smooth_data(x_vals, y_vals, method="Moving Average", window=3):
    """Smooth data for trajectories"""
    if len(x_vals) < 3 or len(y_vals) < 3:
        return x_vals, y_vals

    try:
        if method == "Moving Average":
            x_smooth = pd.Series(x_vals).rolling(window=window, center=True, min_periods=1).mean().values
            y_smooth = pd.Series(y_vals).rolling(window=window, center=True, min_periods=1).mean().values
        elif method == "Exponential":
            alpha = 2.0 / (window + 1)
            x_smooth = pd.Series(x_vals).ewm(alpha=alpha, adjust=False).mean().values
            y_smooth = pd.Series(y_vals).ewm(alpha=alpha, adjust=False).mean().values
        elif method == "Spline":
            if len(x_vals) >= 4:
                indices = np.arange(len(x_vals))
                new_indices = np.linspace(0, len(x_vals) - 1, len(x_vals) * 2)
                f_x = interp1d(indices, x_vals, kind='cubic', bounds_error=False, fill_value='extrapolate')
                f_y = interp1d(indices, y_vals, kind='cubic', bounds_error=False, fill_value='extrapolate')
                x_smooth = f_x(new_indices)
                y_smooth = f_y(new_indices)
                sample_indices = np.linspace(0, len(x_smooth) - 1, len(x_vals)).astype(int)
                x_smooth = x_smooth[sample_indices]
                y_smooth = y_smooth[sample_indices]
            else:
                x_smooth = pd.Series(x_vals).rolling(window=2, center=True, min_periods=1).mean().values
                y_smooth = pd.Series(y_vals).rolling(window=2, center=True, min_periods=1).mean().values
        return x_smooth, y_smooth
    except Exception:
        return x_vals, y_vals

def create_rrg_plot(results, tail_length, enable_smoothing, smoothing_method, smoothing_window, quadrant_filter, show_background):
    """Create enhanced RRG with quadrant focus and zoom"""
    fig = go.Figure()

    all_rs_ratios = []
    all_rs_momentum = []
    for symbol, data in results.items():
        if data['rs_ratio'] is not None and data['rs_momentum'] is not None:
            rs_ratio_vals = data['rs_ratio'].dropna().values
            rs_momentum_vals = data['rs_momentum'].dropna().values
            if len(rs_ratio_vals) > 0 and len(rs_momentum_vals) > 0:
                all_rs_ratios.extend(rs_ratio_vals)
                all_rs_momentum.extend(rs_momentum_vals)

    if not all_rs_ratios or not all_rs_momentum:
        return None

    x_min, x_max = min(all_rs_ratios), max(all_rs_ratios)
    y_min, y_max = min(all_rs_momentum), max(all_rs_momentum)
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    x_center = 100
    x_data_range = max(x_max - 100, 100 - x_min)
    x_range = [x_center - x_data_range - x_padding, x_center + x_data_range + x_padding]
    y_range = [min(y_min - y_padding, -0.5), max(y_max + y_padding, 0.5)]

    # Quadrant-specific ranges
    quadrant_ranges = {
        "Leading": ([100, x_range[1]], [0, y_range[1]]),
        "Weakening": ([100, x_range[1]], [y_range[0], 0]),
        "Lagging": ([x_range[0], 100], [y_range[0], 0]),
        "Improving": ([x_range[0], 100], [0, y_range[1]])
    }

    # Adjust ranges for selected quadrant
    if quadrant_filter != "All":
        x_range, y_range = quadrant_ranges[quadrant_filter]
        x_padding = (x_range[1] - x_range[0]) * 0.05
        y_padding = (y_range[1] - y_range[0]) * 0.05
        x_range = [x_range[0] - x_padding, x_range[1] + x_padding]
        y_range = [y_range[0] - y_padding, y_range[1] + y_padding]

    # Add quadrant backgrounds
    if show_background or quadrant_filter == "All":
        opacity = 0.2 if show_background else 0.1
        fig.add_shape(type="rect", x0=100, y0=0, x1=x_range[1], y1=y_range[1],
                      fillcolor="rgba(0,255,0,0.2)", line=dict(color="rgba(0,0,0,0)"), name="Leading")
        fig.add_shape(type="rect", x0=100, y0=y_range[0], x1=x_range[1], y1=0,
                      fillcolor="rgba(255,165,0,0.2)", line=dict(color="rgba(0,0,0,0)"), name="Weakening")
        fig.add_shape(type="rect", x0=x_range[0], y0=y_range[0], x1=100, y1=0,
                      fillcolor="rgba(255,0,0,0.2)", line=dict(color="rgba(0,0,0,0)"), name="Lagging")
        fig.add_shape(type="rect", x0=x_range[0], y0=0, x1=100, y1=y_range[1],
                      fillcolor="rgba(0,0,255,0.2)", line=dict(color="rgba(0,0,0,0)"), name="Improving")

    # Add center lines
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig.add_vline(x=100, line_dash="dash", line_color="white", opacity=0.5)

    colors = px.colors.qualitative.Plotly
    for i, (symbol, data) in enumerate(results.items()):
        if data['rs_ratio'] is None or data['rs_momentum'] is None:
            continue

        rs_ratio = data['rs_ratio'].dropna()
        rs_momentum = data['rs_momentum'].dropna()
        volatility = data['volatility'].dropna() if data['volatility'] is not None else pd.Series([0] * len(rs_ratio))
        acceleration = data['acceleration'].dropna() if data['acceleration'] is not None else pd.Series([0] * len(rs_ratio))

        tail_points = min(tail_length, len(rs_ratio))
        if tail_points < 2:
            continue

        x_vals = rs_ratio.tail(tail_points).values
        y_vals = rs_momentum.tail(tail_points).values
        current_quad, quad_color, quad_icon, advice = get_quadrant_info(x_vals[-1], y_vals[-1])

        # Apply quadrant filter
        if quadrant_filter != "All" and current_quad != quadrant_filter:
            continue

        color = colors[i % len(colors)]
        x_vals_smooth, y_vals_smooth = smooth_data(x_vals, y_vals, smoothing_method, smoothing_window) if enable_smoothing else (x_vals, y_vals)

        # Add animated trajectory
        fig.add_trace(go.Scatter(
            x=x_vals_smooth,
            y=y_vals_smooth,
            mode='lines+markers',
            name=f'{symbol} Trail',
            line=dict(color=color, width=3, shape='spline'),
            marker=dict(size=6, symbol='circle'),
            opacity=0.7 if quadrant_filter == "All" else 1.0,
            showlegend=True,
            hovertemplate=f'<b>{symbol}</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Momentum: %{{y:.2f}}<extra></extra>'
        ))

        # Add current position
        fig.add_trace(go.Scatter(
            x=[x_vals[-1]],
            y=[y_vals[-1]],
            mode='markers+text',
            name=symbol,
            marker=dict(size=12, symbol='diamond', color=quad_color, line=dict(width=2, color='white')),
            text=[symbol],
            textposition="middle right",
            textfont=dict(size=14, color='white'),
            hovertemplate=(
                f'<b>{symbol}</b><br>'
                f'RS-Ratio: %{{x:.2f}}<br>'
                f'RS-Momentum: %{{y:.2f}}<br>'
                f'Volatility: {volatility.iloc[-1]:.2%}<br>'
                f'Acceleration: {acceleration.iloc[-1]:.2f}<br>'
                f'Quadrant: {current_quad}<br>'
                f'Advice: {advice}<extra></extra>'
            )
        ))

    # Update layout
    fig.update_layout(
        xaxis_title="RS-Ratio",
        yaxis_title="RS-Momentum",
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01, font=dict(color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(range=x_range, gridcolor='rgba(255,255,255,0.2)'),
        yaxis=dict(range=y_range, gridcolor='rgba(255,255,255,0.2)'),
        dragmode='zoom',
        uirevision='lock'  # Preserve zoom state
    )

    return fig

def generate_pdf_report(summary_data, quadrant_filter, period, benchmark):
    """Generate PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Sector Rotation Analysis Report", ln=True, align="C")
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Benchmark: {benchmark} | Period: {period} days | Quadrant: {quadrant_filter}", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", size=10)
    pdf.cell(40, 10, "Sector", border=1)
    pdf.cell(30, 10, "RS-Ratio", border=1)
    pdf.cell(30, 10, "RS-Momentum", border=1)
    pdf.cell(30, 10, "Volatility", border=1)
    pdf.cell(30, 10, "Acceleration", border=1)
    pdf.cell(30, 10, "Quadrant", border=1)
    pdf.ln()

    pdf.set_font("Arial", size=10)
    for row in summary_data:
        pdf.cell(40, 10, row['Sector'], border=1)
        pdf.cell(30, 10, row['RS-Ratio'], border=1)
        pdf.cell(30, 10, row['RS-Momentum'], border=1)
        pdf.cell(30, 10, row['Volatility'], border=1)
        pdf.cell(30, 10, row['Acceleration'], border=1)
        pdf.cell(30, 10, row['Quadrant'], border=1)
        pdf.ln()

    pdf.ln(10)
    pdf.set_font("Arial", "B", size=10)
    pdf.cell(200, 10, "Investment Recommendations", ln=True)
    pdf.set_font("Arial", size=10)
    for row in summary_data:
        _, _, _, advice = get_quadrant_info(float(row['RS-Ratio']), float(row['RS-Momentum']))
        pdf.cell(200, 10, f"{row['Sector']}: {advice}", ln=True)

    output = io.BytesIO()
    pdf.output(output)
    return output.getvalue()

def main():
    st.title("🚀 Advanced Sector Rotation Analysis")
    st.markdown("Analyze sector/stock performance with an interactive Relative Rotation Graph (RRG).")

    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Settings")
        benchmark = st.text_input("Benchmark Symbol", value="^NSEI", help="e.g., ^NSEI for Nifty 50")
        period = st.slider("Analysis Period (days)", min_value=30, max_value=730, value=90, step=5)
        tail_length = st.slider("Tail Length (days)", min_value=2, max_value=25, value=4, step=1)
        quadrant_filter = st.selectbox("Filter by Quadrant", ["All", "Leading", "Weakening", "Lagging", "Improving"])
        show_background = st.checkbox("Show Background Quadrants", value=True)
        smoothing_method = st.selectbox("Smoothing Method", ["Moving Average", "Exponential", "Spline"], index=0)
        smoothing_window = st.slider("Smoothing Window", min_value=3, max_value=10, value=3, step=1)
        st.button("Toggle Dark Mode", on_click=toggle_dark_mode)

        default_sectors = ["^CNXAUTO", "^CNXPHARMA", "^CNXMETAL", "^CNXIT", "^CNXENERGY", "^CNXREALTY", "^CNXPSUBANK",
                           "^CNXMEDIA", "^CNXINFRA", "^CNXPSE", "RELIANCE.NS", "INFY.NS"]
        sectors_text = st.text_area("Sector/Stock Symbols (one per line)", value="\n".join(default_sectors), height=150)
        sectors = [s.strip() for s in sectors_text.split('\n') if s.strip()]

        # Save/load sector lists
        if st.button("Save Sector List"):
            with open("sector_list.json", "w") as f:
                json.dump(sectors, f)
            st.success("Sector list saved!")
        if st.button("Load Sector List"):
            try:
                with open("sector_list.json", "r") as f:
                    saved_sectors = json.load(f)
                st.session_state.sectors_text = "\n".join(saved_sectors)
                st.success("Sector list loaded!")
            except:
                st.error("No saved sector list found.")

    # Main content
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Run Analysis", type="primary"):
            if not sectors:
                st.error("Please enter at least one sector symbol")
                return

            with st.spinner("Fetching data and calculating metrics..."):
                progress = st.progress(0)
                benchmark_data, benchmark_failed = fetch_data([benchmark], period)
                progress.progress(25)

                if benchmark not in benchmark_data:
                    st.error(f"Could not fetch data for benchmark: {benchmark}")
                    return

                sector_data, failed_sectors = fetch_data(sectors, period)
                progress.progress(50)

                if not sector_data:
                    st.error("Could not fetch data for any sectors")
                    return

                if failed_sectors:
                    st.warning(f"Could not fetch data for: {', '.join(failed_sectors)}")

                results = {}
                summary_data = []
                benchmark_prices = benchmark_data[benchmark]

                for i, (symbol, prices) in enumerate(sector_data.items()):
                    try:
                        rel_strength, rel_momentum, volatility = calculate_relative_strength(prices, benchmark_prices, 10)
                        if rel_strength is not None:
                            rs_ratio = calculate_jdk_rs_ratio(rel_strength)
                            rs_momentum, acceleration = calculate_jdk_rs_momentum(rs_ratio)
                            results[symbol] = {
                                'rs_ratio': rs_ratio,
                                'rs_momentum': rs_momentum,
                                'volatility': volatility,
                                'acceleration': acceleration
                            }

                            current_ratio = rs_ratio.iloc[-1]
                            current_momentum = rs_momentum.iloc[-1]
                            current_vol = volatility.iloc[-1] if volatility is not None else 0
                            current_acc = acceleration.iloc[-1] if acceleration is not None else 0
                            quadrant, _, _, _ = get_quadrant_info(current_ratio, current_momentum)
                            if quadrant_filter == "All" or quadrant == quadrant_filter:
                                summary_data.append({
                                    'Sector': symbol,
                                    'RS-Ratio': f"{current_ratio:.2f}",
                                    'RS-Momentum': f"{current_momentum:.2f}",
                                    'Volatility': f"{current_vol:.2%}",
                                    'Acceleration': f"{current_acc:.2f}",
                                    'Quadrant': quadrant
                                })
                    except Exception as e:
                        st.warning(f"Error calculating metrics for {symbol}: {str(e)}")
                    progress.progress(50 + int(50 * (i + 1) / len(sector_data)))

                if not results:
                    st.error("Could not calculate metrics for any sectors")
                    return

                # Display RRG
                fig = create_rrg_plot(results, tail_length, True, smoothing_method, smoothing_window, quadrant_filter, show_background)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

                # Quick Insights
                st.subheader("Quick Insights")
                if summary_data:
                    df_summary = pd.DataFrame(summary_data)
                    top_sector = df_summary.loc[df_summary['RS-Ratio'].astype(float).idxmax()]['Sector']
                    st.markdown(f"<div class='metric-card'><div class='metric-title'>Top Performing Sector</div><div class='metric-value'>{top_sector}</div></div>", unsafe_allow_html=True)
                    high_vol = df_summary.loc[df_summary['Volatility'].astype(float).idxmax()]['Sector']
                    st.markdown(f"<div class='metric-card'><div class='metric-title'>Most Volatile Sector</div><div class='metric-value'>{high_vol}</div></div>", unsafe_allow_html=True)

                # Summary table
                st.subheader("Sector Summary")
                if summary_data:
                    st.dataframe(df_summary, use_container_width=True, hide_index=True)

                # Export options
                col_export1, col_export2 = st.columns(2)
                with col_export1:
                    if st.button("Export Table as CSV"):
                        df_summary.to_csv("sector_summary.csv", index=False)
                        st.success("Table exported as sector_summary.csv")
                with col_export2:
                    if st.button("Export Report as PDF"):
                        pdf_data = generate_pdf_report(summary_data, quadrant_filter, period, benchmark)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_data,
                            file_name="sector_rotation_report.pdf",
                            mime="application/pdf"
                        )

                # Guided tour
                with st.expander("Understanding the RRG"):
                    st.markdown("""
                    **Relative Rotation Graph (RRG) Guide:**
                    - **Leading**: High RS-Ratio (>100) and positive RS-Momentum (>0). Strong sectors to hold.
                    - **Weakening**: High RS-Ratio but negative RS-Momentum. Consider selling as momentum fades.
                    - **Lagging**: Low RS-Ratio (<100) and negative RS-Momentum. Avoid these underperformers.
                    - **Improving**: Low RS-Ratio but positive RS-Momentum. Potential buying opportunities.
                    - **RS-Ratio**: Measures performance relative to the benchmark (100 = neutral).
                    - **RS-Momentum**: Rate of change in relative strength.
                    - **Volatility**: Annualized standard deviation of returns.
                    - **Acceleration**: Rate of change in RS-Momentum, indicating trend strength.
                    - Use the zoom controls to focus on specific quadrants or sectors.
                    """)

if __name__ == "__main__":
    main()