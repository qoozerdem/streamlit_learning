
import streamlit as st
import pandas as pd
import numpy as np

# Generate sample data
dates = pd.date_range(start="2023-01-01", periods=10)
values = np.random.randn(10).cumsum()
data = pd.DataFrame({'Date': dates, 'Value': values})

# Streamlit app
st.title("Basic Line Chart Example")
st.line_chart(data.set_index('Date'))
