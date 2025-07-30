
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go




# Load the CSV file
df = pd.read_csv('output.csv')

# Frequency values
freq = [4435.2, 4531.2, 4646.4, 4742.4, 4838.4, 4934.4, 5203.2, 4761.6, 4800.0, 4857.6, 5107.2]

# Define vmin range
vmin_values = np.linspace(0.8, 1.3, 100)

# Streamlit UI
st.title("Frequency Analysis Dashboard")


# --- CDF Plot Section ---
st.header("CDF Plot for Frequency Columns")
st.write("Select the frequencies and adjust the axis ranges for better visualization.")

# Multiselect widget for frequency selection
selected_freq = st.multiselect("Choose Frequencies (Hz):", freq, default=freq)

# Sliders for vmin (x-axis) and CDF (y-axis) ranges
vmin_min, vmin_max = st.slider("Select vmin Range (X-axis)", 0.8, 1.3, (0.8, 1.3), step=0.01)
cdf_min, cdf_max = st.slider("Select CDF Range (Y-axis)", 0.0, 1.0, (0.0, 1.0), step=0.01)

# Filter vmin values based on slider
vmin_values_filtered = np.linspace(vmin_min, vmin_max, 100)

# Create Plotly figure for CDF
fig_cdf = go.Figure()

for i, f in enumerate(freq):
    if f in selected_freq and i < df.shape[1] - 1:
        column = df.iloc[:, i + 1]
        cdf = [(column <= vmin).sum() / len(column) for vmin in vmin_values_filtered]
        fig_cdf.add_trace(go.Scatter(x=vmin_values_filtered, y=cdf, mode='lines', name=f'{f} Hz'))

fig_cdf.update_layout(
    title='CDF',
    xaxis_title='vmin',
    yaxis_title='CDF (Proportion below vmin)',
    xaxis=dict(range=[vmin_min, vmin_max]),
    yaxis=dict(range=[cdf_min, cdf_max]),
    legend_title='Frequency',
    template='plotly_white'
)

# Display the CDF plot
st.plotly_chart(fig_cdf)


# --- mV/MHz Plot Section ---
st.header("mV/MHz")

mv_per_mhz = []
valid_freq = []




for i, f in enumerate(freq):
    if i < df.shape[1] - 1:
        column = df.iloc[:, i + 1]
        avg_mv = column.mean()-column.min()
        mv_mhz = (avg_mv * 1000) / (f-4435.2)  # Convert Hz to MHz and compute mV/MHz
        mv_per_mhz.append(mv_mhz)
        valid_freq.append(f)

# Sort the data by frequency
sorted_data = sorted(zip(valid_freq, mv_per_mhz))
sorted_freq, sorted_mv_per_mhz = zip(*sorted_data)

# Create Plotly figure with frequency on x-axis
fig_mv_mhz = go.Figure()
fig_mv_mhz.add_trace(go.Scatter(
    x=sorted_freq,
    y=sorted_mv_per_mhz,
    mode='lines+markers',
    name='mV/MHz',
    line=dict(shape='linear')
))

# Define regular tick intervals for the x-axis
min_freq = min(sorted_freq)
max_freq = max(sorted_freq)
tick_interval = (max_freq - min_freq) // 10 if max_freq > min_freq else 1
tick_vals = list(np.arange(min_freq, max_freq + tick_interval, tick_interval))

fig_mv_mhz.update_layout(
    title='mV/MHz',
    xaxis_title='Frequency (Hz)',
    yaxis_title='mV/MHz',
    xaxis=dict(tickmode='array', tickvals=tick_vals),
    template='plotly_white'
)

# Display the plot
st.plotly_chart(fig_mv_mhz)

# --- Gauge Chart Section ---
st.header("Yield Gauge")

# Sliders for threshold and adder
threshold = st.slider("Select Threshold Value", min_value=0.8, max_value=1.3, value=1.0, step=0.001)
adder = st.slider("Select Adder Value", min_value=0.0, max_value=0.5, value=0.1, step=0.01)

# Compute yield for selected frequencies
yields = []
for i, f in enumerate(freq):
    if f in selected_freq and i < df.shape[1] - 1:
        column = df.iloc[:, i + 1]
        cutoff = threshold - adder
        yield_percent = (column <= cutoff).sum() / len(column) * 100
        yields.append(yield_percent)

# Compute average yield
average_yield = np.mean(yields) if yields else 0

# Create gauge chart
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=average_yield,
    title={'text': "Average Yield (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 50], 'color': "red"},
            {'range': [50, 80], 'color': "yellow"},
            {'range': [80, 100], 'color': "green"}
        ]
    }
))

# Display the gauge chart
st.plotly_chart(fig_gauge)
