import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ========================= Custom CSS for Compact Controls =========================
st.markdown(
    """
    <style>
    .element-container, .stSlider, .stSelectbox, .stMultiSelect, .stDataFrame, .stTable {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }
    .stSlider > div { padding-top: 0rem !important; }
    .stPlotlyChart { margin-top: 0rem !important; }
    .stDataFrame { margin-bottom: 0rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================= Load Data =========================
df = pd.read_csv('output.csv')
freq = [4435.2, 4531.2, 4646.4, 4742.4, 4838.4, 4934.4, 5203.2, 4761.6, 4800.0, 4857.6, 5107.2]
vmin_values = np.linspace(0.8, 1.3, 100)

# ========================= Page Title =========================
st.set_page_config(layout="wide")

# ========================= CDF Plot Section =========================
with st.container():
    maincol1, maincol2 = st.columns([3, 1])

    with maincol1:
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            selected_freq = st.multiselect(
                "Frequencies (MHz):", freq, default=freq
            )

        with col2:
            threshold = st.slider(
                "Threshold",
                min_value=0.8,
                max_value=1.3,
                value=1.08,
                step=0.001,
                key="threshold_slider"
            )
            adder = st.slider(
                "Adder",
                min_value=0.0,
                max_value=0.5,
                value=0.0,
                step=0.01,
                key="adder_slider"
            )

        with col3:
            yields = []
            yield_col1, yield_col2 = st.columns([3, 2])
            with yield_col1:
                for i, f in enumerate(freq):
                    if f in selected_freq and i < df.shape[1] - 1:
                        column = df.iloc[:, i + 1]
                        cutoff = threshold - adder
                        yield_percent = (column[column != 1.3] <= cutoff).sum() / len(column) * 100
                        yields.append(yield_percent)
                average_yield = np.mean(yields) if yields else 0

                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=average_yield,
                        title={'text': "Avg Yield (%)", 'font': {'size': 16}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "red"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"},
                            ],
                        },
                        domain={'x': [0, 1], 'y': [0, 1]}
                    )
                )
                fig_gauge.update_layout(margin=dict(t=10, b=0), height=200)
                st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Fmax vs Yield")

        fig_cdf = go.Figure()
        for i, f in enumerate(freq):
          if f in selected_freq and i < df.shape[1] - 1:
              column = df.iloc[:, i + 1]
              cdf = [
                (column[column != 1.3] <= vmin - adder).sum() / len(column) * 100
                for vmin in vmin_values
            ]
              fig_cdf.add_trace(
                go.Scatter(
                    x=vmin_values,
                    y=cdf,
                    mode='lines+markers',
                    name=f'{f} MHz',
                    hovertemplate='Vmin: %{x:.3f}<br>yield: %{y:.2f}%<extra>%{fullData.name}</extra>'
                )
            )

        fig_cdf.update_layout(
        xaxis_title='vmin',
        yaxis_title='CDF (%)',
        xaxis=dict(range=[0.8, 1.3]),
        yaxis=dict(range=[0, 110]),
        legend_title='Frequency',
        template='plotly_white',
        height=600,
        hovermode='x unified',
    )
        st.plotly_chart(fig_cdf, use_container_width=True)

    with maincol2:
        freq_yield_data = []
        for i, f in enumerate(freq):
            if i < df.shape[1] - 1:
                column = df.iloc[:, i + 1]
                cutoff = threshold - adder
                yield_percent = (column[column != 1.3] <= cutoff).sum() / len(column) * 100
                freq_yield_data.append({
                    "Freq": f"{f:.2f}",
                    "Yield (%)": f"{yield_percent:.2f}",
                })
        if freq_yield_data:
            df_yields = pd.DataFrame(freq_yield_data)
            st.dataframe(df_yields, use_container_width=False, height=450)
        else:
            st.info("No freq selected.")

        mv_per_mhz = []
        valid_freq = []

        freqmin = min(freq)
        v_min_mean = []
        for i in range(11):
            mean_val = df.iloc[:, i + 1].mean()
            if mean_val != 1.30:
                v_min_mean.append(df.iloc[:, i + 1][df.iloc[:, i + 1] != 1.3].mean())
        v_min_mean = min(v_min_mean) if v_min_mean else 0

        for i, f in enumerate(freq):
            column = df.iloc[:, i + 1]
            mean_val = column.mean()
            if i == 0 or mean_val == 1.3 or len(column[column != 1.3]) < 26:
                continue
            if i < df.shape[1] - 1:
                column = column[column != 1.3]
                avg_mv = column.mean() - v_min_mean
                mv_mhz = (f - freqmin) / (avg_mv * 1000) if avg_mv != 0 else 0
                mv_per_mhz.append(mv_mhz)
                valid_freq.append(f)

        if valid_freq and mv_per_mhz:
            sorted_data = sorted(zip(valid_freq, mv_per_mhz))
            sorted_freq, sorted_mv_per_mhz = zip(*sorted_data)

            fig_mv_mhz = go.Figure()
            fig_mv_mhz.add_trace(
                go.Scatter(
                    x=sorted_freq,
                    y=sorted_mv_per_mhz,
                    mode='lines+markers',
                    name='MHz per mV',
                    line=dict(shape='linear'),
                )
            )

            tick_interval = (max(sorted_freq) - min(sorted_freq)) // 10 or 1
            tick_vals = list(np.arange(min(sorted_freq), max(sorted_freq) + tick_interval, tick_interval))

            fig_mv_mhz.update_layout(
                title='MHz per mV',
                xaxis_title='Frequency (MHz)',
                yaxis_title='MHz/mV',
                xaxis=dict(tickmode='array', tickvals=tick_vals),
                template='plotly_white',
                height=450,
            )
            st.plotly_chart(fig_mv_mhz, use_container_width=True)