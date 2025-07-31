import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ========================= Load Data =========================
df = pd.read_csv('output.csv')
freq = [4435.2, 4531.2, 4646.4, 4742.4, 4838.4, 4934.4, 5203.2, 4761.6, 4800.0, 4857.6, 5107.2]
vmin_values = np.linspace(0.8, 1.3, 100)

# ========================= Page Title =========================
st.set_page_config(layout="wide")

# ========================= CDF Plot Section =========================
with st.container():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Fmax vs Yield")
        selected_freq = st.multiselect("Choose Frequencies (MHz):", freq, default=freq)

        fig_cdf = go.Figure()

        for i, f in enumerate(freq):
            if f in selected_freq and i < df.shape[1] - 1:
                column = df.iloc[:, i + 1]
                cdf = [(column <= vmin).sum() / len(column) * 100 for vmin in vmin_values]
                fig_cdf.add_trace(go.Scatter(
                    x=vmin_values,
                    y=cdf,
                    mode='lines+markers',
                    name=f'{f} MHz',
                    hovertemplate='Vmin: %{x:.3f}<br>yield: %{y:.2f}%<extra>%{fullData.name}</extra>'
                ))

        fig_cdf.update_layout(
            xaxis_title='vmin',
            yaxis_title='CDF (%)',
            xaxis=dict(range=[0.8, 1.3]),
            yaxis=dict(range=[0, 100]),
            legend_title='Frequency',
            template='plotly_white',
            height=600,
            hovermode='x unified',
        )

        st.plotly_chart(fig_cdf, use_container_width=True)

    with col2:
        mv_per_mhz = []
        valid_freq = []

        freqmin= min(freq)
        v_min_mean=[]
        for i in range(11):
            if df.iloc[:, i + 1].mean() != 1.30:
                v_min_mean.append(df.iloc[:, i + 1][df.iloc[:, i + 1]!=1.3].mean())
        v_min_mean=min(v_min_mean)

        for i, f in enumerate(freq):
            if i == 0 or df.iloc[:, i + 1].mean() == 1.3 or len(df.iloc[:, i + 1][df.iloc[:, i + 1]!=1.3])<26 :
                continue
            if i < df.shape[1] - 1:
                column = df.iloc[:, i + 1][df.iloc[:, i + 1]!=1.3]
                avg_mv = column.mean() - v_min_mean
                mv_mhz = (f - freqmin) / (avg_mv * 1000) 
                mv_per_mhz.append(mv_mhz)
                valid_freq.append(f)

        sorted_data = sorted(zip(valid_freq, mv_per_mhz))
        sorted_freq, sorted_mv_per_mhz = zip(*sorted_data)

        fig_mv_mhz = go.Figure()
        fig_mv_mhz.add_trace(go.Scatter(
            x=sorted_freq,
            y=sorted_mv_per_mhz,
            mode='lines+markers',
            name='MHz per mV',
            line=dict(shape='linear')))

        tick_interval = (max(sorted_freq) - min(sorted_freq)) // 10 or 1
        tick_vals = list(np.arange(min(sorted_freq), max(sorted_freq) + tick_interval, tick_interval))

        fig_mv_mhz.update_layout(
            title='MHz per mV',
            xaxis_title='Frequency (MHz)',
            yaxis_title='MHz/mV',
            xaxis=dict(tickmode='array', tickvals=tick_vals),
            template='plotly_white',
            height=350)

        st.plotly_chart(fig_mv_mhz, use_container_width=True)

        # Threshold and Adder Sliders (smaller font and tighter layout)
        yields = []
        freq_yield_data = []  # <-- New: to store (frequency, yield) tuples
                
        gaugecol1, gaugecol2,gaugecol3 = st.columns([1,2,3])
        with gaugecol1:
        
             threshold = st.slider("Threshold Value", min_value=0.8, max_value=1.3, value=1.08, step=0.001,key="threshold_slider",width=100)

             adder = st.slider( "Adder Value", min_value=0.0, max_value=0.5, value=0.0, step=0.01,key="adder_slider",width=100)
        with gaugecol2:

         for i, f in enumerate(freq):
            if f in selected_freq and i < df.shape[1] - 1:
                column = df.iloc[:, i + 1]
                cutoff = threshold - adder
                yield_percent = (column <= cutoff).sum() / len(column) * 100
                yields.append(yield_percent)
                freq_yield_data.append((f, yield_percent))  # <-- Save per-frequency

         average_yield = np.mean(yields) if yields else 0

        # ------ Yield per Frequency Table ------
         if freq_yield_data:
            df_yield = pd.DataFrame(freq_yield_data, columns=["Frequency (MHz)", "Yield (%)"])
            df_yield["Frequency (MHz)"] = df_yield["Frequency (MHz)"].map('{:.1f}'.format)
            df_yield["Yield (%)"] = df_yield["Yield (%)"].map('{:.2f}'.format)

            st.table(df_yield)
         else:
            st.info("No frequency selected for yield calculation.")
        with gaugecol3:
         

        # ------ Yield Gauge (keep your code as-is) ------
         fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=average_yield,
            title={'text': "Average Yield (%)", 'font': {'size': 18}},
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
        ))
         fig_gauge.update_layout(margin=dict(t=30, b=10), height=350)
         st.plotly_chart(fig_gauge, use_container_width=True)