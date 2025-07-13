import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Enhanced Sector Rotation Analysis",
    layout="centered"
)

# Apply fixed screen width for app (1440px)
st.markdown(
    f"""
    <style>
      .stAppViewContainer .stMain .stMainBlockContainer{{ max-width: 1440px; }}
      .metric-card {{
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
      }}
      .quadrant-info {{
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem;
        text-align: center;
      }}
      .leading {{ background: rgba(76,175,80,0.15); border: 2px solid #4CAF50; }}
      .weakening {{ background: rgba(255,152,0,0.15); border: 2px solid #FF9800; }}
      .lagging {{ background: rgba(244,67,54,0.15); border: 2px solid #F44336; }}
      .improving {{ background: rgba(33,150,243,0.15); border: 2px solid #2196F3; }}
    </style>    
    """,
    unsafe_allow_html=True,
)

def fetch_data(symbols, period_days):
    """Fetch data from Yahoo Finance with error handling"""
    data = {}
    failed_symbols = []

    # Calculate start date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days + 10)  # Add buffer for calculations

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols):
        try:
            status_text.text(f"Fetching data for {symbol}...")
            progress_bar.progress((i + 1) / len(symbols))
            
            hist = yf.download(symbol, start_date, end_date, progress=False, auto_adjust=True, prepost=True, threads=True)

            if len(hist) < 20:  # Minimum data requirement
                failed_symbols.append(f"{symbol} (Insufficient data)")
                continue

            # Get close prices - handle different column formats
            if 'Close' in hist.columns:
                close_prices = hist['Close']
            elif 'Adj Close' in hist.columns:
                close_prices = hist['Adj Close']
            else:
                close_prices = hist.iloc[:, -1]  # Last column as fallback
                
            # Clean the data
            close_prices = close_prices.dropna()
            
            if len(close_prices) < 20:
                failed_symbols.append(f"{symbol} (Too many missing values)")
                continue
                
            data[symbol] = close_prices
            
        except Exception as e:
            failed_symbols.append(f"{symbol} (Error: {str(e)[:50]})")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return data, failed_symbols


def calculate_relative_strength(price_data, benchmark_data, period):
    """Calculate relative strength vs benchmark"""
    # Ensure we have enough data
    min_length = min(len(price_data), len(benchmark_data))
    if min_length < period:
        return None, None

    # Align data by index
    aligned_data = pd.DataFrame({
        'price': price_data,
        'benchmark': benchmark_data
    }).dropna()

    if len(aligned_data) < period:
        return None, None

    # Calculate relative strength (sector/benchmark)
    relative_strength = aligned_data['price'] / aligned_data['benchmark']

    # Calculate momentum (rate of change)
    rs_momentum = relative_strength.pct_change(period).dropna()

    return relative_strength, rs_momentum


def calculate_jdk_rs_ratio(relative_strength, short_period=10, long_period=40):
    """Calculate JdK RS-Ratio similar to RRG methodology"""
    if len(relative_strength) < long_period:
        return None

    # Normalize relative strength to 100
    rs_normalized = (relative_strength / relative_strength.rolling(long_period).mean()) * 100

    return rs_normalized


def calculate_jdk_rs_momentum(rs_ratio, period=10):
    """Calculate JdK RS-Momentum"""
    if rs_ratio is None or len(rs_ratio) < period:
        return None

    # Calculate momentum as rate of change
    momentum = ((rs_ratio / rs_ratio.shift(period)) - 1) * 100

    return momentum


def get_quadrant_info(rs_ratio, rs_momentum):
    """Determine quadrant and provide info"""
    if rs_ratio > 100 and rs_momentum > 0:
        return "Leading", "#4CAF50", "🚀", "Strong outperformance with positive momentum"
    elif rs_ratio > 100 and rs_momentum < 0:
        return "Weakening", "#FF9800", "📉", "Outperforming but losing momentum"
    elif rs_ratio < 100 and rs_momentum < 0:
        return "Lagging", "#F44336", "📊", "Underperforming with negative momentum"
    else:
        return "Improving", "#2196F3", "📈", "Underperforming but gaining momentum"


def smooth_data(x_vals, y_vals, method="Moving Average", window=3):
    """Smooth the tail data using various methods"""
    if len(x_vals) < 3 or len(y_vals) < 3:
        return x_vals, y_vals

    try:
        if method == "Moving Average":
            # Simple moving average
            x_smooth = pd.Series(x_vals).rolling(window=window, center=True, min_periods=1).mean().values
            y_smooth = pd.Series(y_vals).rolling(window=window, center=True, min_periods=1).mean().values

        elif method == "Exponential":
            # Exponential smoothing
            alpha = 2.0 / (window + 1)
            x_smooth = pd.Series(x_vals).ewm(alpha=alpha, adjust=False).mean().values
            y_smooth = pd.Series(y_vals).ewm(alpha=alpha, adjust=False).mean().values

        elif method == "Spline":
            # Spline interpolation for smoothing
            if len(x_vals) >= 4:  # Need at least 4 points for cubic spline
                indices = np.arange(len(x_vals))

                # Create more points for smoother curve
                new_indices = np.linspace(0, len(x_vals) - 1, len(x_vals) * 2)

                # Interpolate
                f_x = interp1d(indices, x_vals, kind='cubic', bounds_error=False, fill_value='extrapolate')
                f_y = interp1d(indices, y_vals, kind='cubic', bounds_error=False, fill_value='extrapolate')

                x_smooth = f_x(new_indices)
                y_smooth = f_y(new_indices)

                # Sample back to original length but smoothed
                sample_indices = np.linspace(0, len(x_smooth) - 1, len(x_vals)).astype(int)
                x_smooth = x_smooth[sample_indices]
                y_smooth = y_smooth[sample_indices]
            else:
                # Fall back to moving average for short series
                x_smooth = pd.Series(x_vals).rolling(window=2, center=True, min_periods=1).mean().values
                y_smooth = pd.Series(y_vals).rolling(window=2, center=True, min_periods=1).mean().values

        return x_smooth, y_smooth

    except Exception as e:
        # If smoothing fails, return original data
        return x_vals, y_vals


def create_rrg_plot(results, tail_length, enable_smoothing=True, smoothing_method="Moving Average",
                    smoothing_window=3, show_tail=False, active_quadrants=None):
    """Create the Relative Rotation Graph with quadrant filtering"""
    if active_quadrants is None:
        active_quadrants = ["Leading", "Weakening", "Lagging", "Improving"]
    
    fig = go.Figure()

    # First, determine the actual data ranges
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
        st.error("No valid data to plot")
        return None

    # Calculate dynamic ranges with some padding
    x_min, x_max = min(all_rs_ratios), max(all_rs_ratios)
    y_min, y_max = min(all_rs_momentum), max(all_rs_momentum)

    # Add padding (10% on each side)
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1

    x_range = [x_min - x_padding, x_max + x_padding]
    y_range = [y_min - y_padding, y_max + y_padding]

    # Ensure 100 is visible on x-axis and 0 is visible on y-axis
    x_center = 100
    x_data_range = max(x_max - 100, 100 - x_min)  # Get the larger distance from 100
    x_range = [x_center - x_data_range - x_padding, x_center + x_data_range + x_padding]
    if y_range[0] > 0:
        y_range[0] = min(y_range[0], -0.5)
    if y_range[1] < 0:
        y_range[1] = max(y_range[1], 0.5)

    # Add quadrant backgrounds based on actual ranges and active status
    quad_colors = {
        "Leading": "rgba(76,175,80,0.1)" if "Leading" in active_quadrants else "rgba(76,175,80,0.03)",
        "Weakening": "rgba(255,152,0,0.1)" if "Weakening" in active_quadrants else "rgba(255,152,0,0.03)",
        "Lagging": "rgba(244,67,54,0.1)" if "Lagging" in active_quadrants else "rgba(244,67,54,0.03)",
        "Improving": "rgba(33,150,243,0.1)" if "Improving" in active_quadrants else "rgba(33,150,243,0.03)"
    }

    fig.add_shape(
        type="rect",
        x0=100, y0=0, x1=x_range[1], y1=y_range[1],
        fillcolor=quad_colors["Leading"],
        line=dict(color="rgba(0,0,0,0)"),
        name="Leading"
    )
    fig.add_shape(
        type="rect",
        x0=100, y0=y_range[0], x1=x_range[1], y1=0,
        fillcolor=quad_colors["Weakening"],
        line=dict(color="rgba(0,0,0,0)"),
        name="Weakening"
    )
    fig.add_shape(
        type="rect",
        x0=x_range[0], y0=y_range[0], x1=100, y1=0,
        fillcolor=quad_colors["Lagging"],
        line=dict(color="rgba(0,0,0,0)"),
        name="Lagging"
    )
    fig.add_shape(
        type="rect",
        x0=x_range[0], y0=0, x1=100, y1=y_range[1],
        fillcolor=quad_colors["Improving"],
        line=dict(color="rgba(0,0,0,0)"),
        name="Improving"
    )

    # Add center lines
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.add_vline(x=100, line_dash="dash", line_color="black", opacity=0.5)

    colors = px.colors.qualitative.Plotly

    for i, (symbol, data) in enumerate(results.items()):
        if data['rs_ratio'] is None or data['rs_momentum'] is None:
            continue

        rs_ratio = data['rs_ratio'].dropna()
        rs_momentum = data['rs_momentum'].dropna()

        # Get the last 'tail_length' points
        tail_points = min(tail_length, len(rs_ratio))

        if tail_points < 2:
            continue

        x_vals = rs_ratio.tail(tail_points).values
        y_vals = rs_momentum.tail(tail_points).values

        # Get current quadrant
        current_quad, quad_color, quad_icon, quad_desc = get_quadrant_info(x_vals[-1], y_vals[-1])
        
        # Skip if not in active quadrants
        if current_quad not in active_quadrants:
            continue

        # Apply smoothing if enabled
        if enable_smoothing and tail_points > 2:
            x_vals_smooth, y_vals_smooth = smooth_data(x_vals, y_vals, smoothing_method, smoothing_window)
        else:
            x_vals_smooth, y_vals_smooth = x_vals, y_vals

        color = colors[i % len(colors)]

        # Add tail (trajectory) - use smoothed data for the line, original for markers
        if show_tail:
            fig.add_trace(go.Scatter(
                x=x_vals_smooth,
                y=y_vals_smooth,
                mode='lines',
                name=f'{symbol} Trail',
                line=dict(color=color, width=3, shape='spline' if smoothing_method == "Spline" else 'linear'),
                opacity=0.7,
                showlegend=False,
                hovertemplate=f'<b>{symbol} Trail</b><extra></extra>'
            ))

            # Add direction markers along the trail
            if len(x_vals) > 3:
                marker_indices = np.linspace(0, len(x_vals_smooth)-1, min(5, len(x_vals_smooth))).astype(int)
                
                # Add arrows to show direction
                for j in range(len(marker_indices)-1):
                    idx = marker_indices[j]
                    next_idx = marker_indices[j+1]
                    
                    fig.add_annotation(
                        x=x_vals_smooth[next_idx],
                        y=y_vals_smooth[next_idx],
                        ax=x_vals_smooth[idx],
                        ay=y_vals_smooth[idx],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1.5,
                        arrowcolor=color,
                        opacity=0.7
                    )

        # Add current position (larger marker) - always use original data
        fig.add_trace(go.Scatter(
            x=[x_vals[-1]],
            y=[y_vals[-1]],
            mode='markers+text',
            name=f'{symbol} ({current_quad})',
            marker=dict(
                size=20,
                color=color,
                line=dict(width=2, color='white'),
                symbol='diamond'
            ),
            text=[f'{symbol}'],
            textposition="middle right",
            textfont=dict(size=15, color='black'),
            hovertemplate=f'<b>{symbol}</b><br>' +
                          f'RS-Ratio: {x_vals[-1]:.2f}<br>' +
                          f'RS-Momentum: {y_vals[-1]:.2f}<br>' +
                          f'Quadrant: {current_quad}<br>' +
                          f'{quad_desc}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title="Relative Rotation Graph (RRG)",
        xaxis_title="RS-Ratio",
        yaxis_title="RS-Momentum",
        width=800,
        height=800,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, b=50, t=80, pad=10)
    )

    # Set dynamic axis ranges
    fig.update_xaxes(range=x_range, gridcolor='lightgray', gridwidth=0.5)
    fig.update_yaxes(range=y_range, gridcolor='lightgray', gridwidth=0.5)

    # Add annotations for quadrants - only show active ones
    x_mid = (x_range[0] + x_range[1]) / 2
    y_mid = (y_range[0] + y_range[1]) / 2

    # Leading quadrant (top-right)
    if "Leading" in active_quadrants:
        leading_x = (100 + x_range[1]) / 2
        leading_y = y_range[1] * 0.8
        fig.add_annotation(x=leading_x, y=leading_y, text="🚀 Leading<br><i>Hold Position</i>",
                           showarrow=False, font=dict(size=14, color="#4CAF50"))

    # Weakening quadrant (bottom-right)
    if "Weakening" in active_quadrants:
        weakening_x = (100 + x_range[1]) / 2
        weakening_y = y_range[0] * 0.8
        fig.add_annotation(x=weakening_x, y=weakening_y, text="📉 Weakening<br><i>Look to Sell</i>",
                           showarrow=False, font=dict(size=14, color="#FF9800"))

    # Lagging quadrant (bottom-left)
    if "Lagging" in active_quadrants:
        lagging_x = (x_range[0] + 100) / 2
        lagging_y = y_range[0] * 0.8
        fig.add_annotation(x=lagging_x, y=lagging_y, text="📊 Lagging<br><i>Avoid</i>",
                           showarrow=False, font=dict(size=14, color="#F44336"))

    # Improving quadrant (top-left)
    if "Improving" in active_quadrants:
        improving_x = (x_range[0] + 100) / 2
        improving_y = y_range[1] * 0.8
        fig.add_annotation(x=improving_x, y=improving_y, text="📈 Improving<br><i>Look to Buy</i>",
                           showarrow=False, font=dict(size=14, color="#2196F3"))

    return fig


def create_summary_table(results, active_quadrants=None):
    """Create summary table with quadrant filtering"""
    if active_quadrants is None:
        active_quadrants = ["Leading", "Weakening", "Lagging", "Improving"]
    
    summary_data = []

    for symbol, data in results.items():
        if data['rs_ratio'] is not None and data['rs_momentum'] is not None:
            current_ratio = data['rs_ratio'].iloc[-1] if len(data['rs_ratio']) > 0 else 0
            current_momentum = data['rs_momentum'].iloc[-1] if len(data['rs_momentum']) > 0 else 0

            quadrant, color, icon, description = get_quadrant_info(current_ratio, current_momentum)
            
            # Skip if not in active quadrants
            if quadrant not in active_quadrants:
                continue

            # Calculate trends
            ratio_trend = "↗️" if len(data['rs_ratio']) > 1 and data['rs_ratio'].iloc[-1] > data['rs_ratio'].iloc[-2] else "↘️"
            momentum_trend = "↗️" if len(data['rs_momentum']) > 1 and data['rs_momentum'].iloc[-1] > data['rs_momentum'].iloc[-2] else "↘️"
            
            summary_data.append({
                'Symbol': f"{icon} {symbol}",
                'RS-Ratio': f"{current_ratio:.2f} {ratio_trend}",
                'RS-Momentum': f"{current_momentum:.2f} {momentum_trend}",
                'Quadrant': quadrant,
                'Signal': description
            })

    return pd.DataFrame(summary_data)


def main():
    st.title("Enhanced Sector Rotation Analysis")
    st.markdown("Analyze sector/stock performance relative to benchmark using RRG methodology")

    # Initialize session state for quadrant filters
    if 'active_quadrants' not in st.session_state:
        st.session_state.active_quadrants = ["Leading", "Weakening", "Lagging", "Improving"]

    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Benchmark input
        benchmark = st.text_input("Benchmark Symbol", value="^NSEI",
                                help="Enter benchmark symbol (e.g., ^NSEI for Nifty 50)")

        # Period slider
        period = st.slider("Analysis Period (days)", min_value=65, max_value=365, value=90, step=5)

        # Tail length slider
        tail_length = st.slider("Tail Length (days)", min_value=2, max_value=25, value=10, step=1)

        # Smoothing options
        st.subheader("Visualization Options")
        smoothing_method = st.selectbox(
            "Smoothing Method",
            ["Moving Average", "Exponential", "Spline"],
            help="Algorithm for smoothing the tail curves"
        )
        
        smoothing_window = st.slider(
            "Smoothing Window",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            help="Window size for smoothing algorithm"
        )
        
        show_tail = st.checkbox("Show Tail", value=True)

    with col2:
        # Sectors input
        default_sectors = ["^CNXAUTO", "^CNXPHARMA", "^CNXMETAL", "^CNXIT", "^CNXENERGY", "^CNXREALTY", "^CNXPSUBANK",
                         "^CNXMEDIA", "^CNXINFRA", "^CNXPSE", "RELIANCE.NS", "INFY.NS"]

        sectors_text = st.text_area(
            "Enter Sector/Stock symbols (one per line)",
            value="\n".join(default_sectors),
            height=220,
            help="Enter each sector/stock symbol on a new line"
        )

        sectors = [s.strip() for s in sectors_text.split('\n') if s.strip()]

        # Quadrant filtering
        st.subheader("Quadrant Filters")
        st.markdown("Select which quadrants to display:")
        
        leading = st.checkbox("🚀 Leading", value="Leading" in st.session_state.active_quadrants, key="leading_check")
        weakening = st.checkbox("📉 Weakening", value="Weakening" in st.session_state.active_quadrants, key="weakening_check")
        lagging = st.checkbox("📊 Lagging", value="Lagging" in st.session_state.active_quadrants, key="lagging_check")
        improving = st.checkbox("📈 Improving", value="Improving" in st.session_state.active_quadrants, key="improving_check")
        
        # Update active quadrants based on checkboxes
        st.session_state.active_quadrants = []
        if leading: st.session_state.active_quadrants.append("Leading")
        if weakening: st.session_state.active_quadrants.append("Weakening")
        if lagging: st.session_state.active_quadrants.append("Lagging")
        if improving: st.session_state.active_quadrants.append("Improving")

    # Analysis button
    if st.button("Run Analysis", type="primary"):
        if not sectors:
            st.error("Please enter at least one sector symbol")
            return

        with st.spinner("Fetching data and calculating metrics..."):
            # Fetch benchmark data
            benchmark_data, benchmark_failed = fetch_data([benchmark], period)

            if benchmark not in benchmark_data:
                st.error(f"Could not fetch data for benchmark: {benchmark}")
                if benchmark_failed:
                    st.error(f"Error details: {benchmark_failed[0]}")
                return

            # Fetch sector data
            sector_data, failed_sectors = fetch_data(sectors, period)

            if not sector_data:
                st.error("Could not fetch data for any sectors")
                return

            # Show warnings for failed symbols
            if failed_sectors:
                with st.expander(f"⚠️ Warning: {len(failed_sectors)} symbols failed"):
                    for failed in failed_sectors:
                        st.warning(f"• {failed}")

            # Calculate relative rotation metrics
            results = {}
            benchmark_prices = benchmark_data[benchmark]

            for symbol, prices in sector_data.items():
                try:
                    # Calculate relative strength
                    rel_strength, rel_momentum = calculate_relative_strength(prices, benchmark_prices, 10)

                    if rel_strength is not None:
                        # Calculate JdK RS-Ratio and RS-Momentum
                        rs_ratio = calculate_jdk_rs_ratio(rel_strength)
                        rs_momentum = calculate_jdk_rs_momentum(rs_ratio)

                        results[symbol] = {
                            'rs_ratio': rs_ratio,
                            'rs_momentum': rs_momentum,
                            'relative_strength': rel_strength
                        }

                except Exception as e:
                    st.warning(f"Error calculating metrics for {symbol}: {str(e)}")
                    continue

            if not results:
                st.error("Could not calculate metrics for any sectors")
                return

            # Create and display the plot
            fig = create_rrg_plot(
                results, 
                tail_length, 
                enable_smoothing=True,
                smoothing_method=smoothing_method,
                smoothing_window=smoothing_window,
                show_tail=show_tail,
                active_quadrants=st.session_state.active_quadrants
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not create RRG plot")
                return

            # Summary table
            st.subheader("Relative Positions of Sector/Stock")
            summary_df = create_summary_table(results, st.session_state.active_quadrants)
            
            if not summary_df.empty:
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No symbols match the current quadrant filters")

            # Quadrant distribution
            if st.session_state.active_quadrants:
                cols = st.columns(len(st.session_state.active_quadrants))
                
                quadrant_counts = summary_df['Quadrant'].value_counts()
                
                for i, quadrant in enumerate(st.session_state.active_quadrants):
                    with cols[i]:
                        count = quadrant_counts.get(quadrant, 0)
                        
                        # Get quadrant info
                        if quadrant == "Leading":
                            color = "#4CAF50"
                            icon = "🚀"
                        elif quadrant == "Weakening":
                            color = "#FF9800"
                            icon = "📉"
                        elif quadrant == "Lagging":
                            color = "#F44336"
                            icon = "📊"
                        else:  # Improving
                            color = "#2196F3"
                            icon = "📈"
                        
                        st.markdown(
                            f'<div class="quadrant-info" style="border-color: {color}">'
                            f'<h3>{icon} {quadrant}</h3><h2>{count}</h2></div>', 
                            unsafe_allow_html=True
                        )

    # Explanation
    with st.expander("📚 Understanding the Relative Rotation Graph"):
        st.markdown("""
            **Quadrants Explanation:**

            **🚀 Leading (Top-Right)**: High relative strength, positive momentum  
            - Sectors out performing benchmark with increasing momentum  
            - **Strategy:** Hold positions, strong performance continues  

            **📉 Weakening (Bottom-Right)**: High relative strength, negative momentum  
            - Sectors still out performing but losing momentum  
            - **Strategy:** Consider taking profits, watch for rotation  

            **📊 Lagging (Bottom-Left)**: Low relative strength, negative momentum  
            - Sectors under performing benchmark with decreasing momentum  
            - **Strategy:** Avoid or reduce exposure  

            **📈 Improving (Top-Left)**: Low relative strength, positive momentum  
            - Sectors under performing but gaining momentum  
            - **Strategy:** Watch for entry opportunities  

            **How to Read:**  
            - **RS-Ratio > 100**: Sector out performing benchmark  
            - **RS-Ratio < 100**: Sector under performing benchmark  
            - **RS-Momentum > 0**: Relative strength is improving  
            - **RS-Momentum < 0**: Relative strength is declining  
            - **Tail:** Shows the trajectory of sector movement over time  
            - **Direction Arrows:** Indicate movement direction along the tail  
            """)


if __name__ == "__main__":
    main()