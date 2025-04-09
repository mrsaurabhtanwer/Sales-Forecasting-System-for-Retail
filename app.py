import streamlit as st
import pandas as pd

# Try importing Prophet
try:
     from prophet import Prophet
except ImportError:
     st.error("Prophet library is not installed. Please install it using 'pip install prophet'.")

st.title("Retail Sales Forecasting App")

uploaded_file = st.file_uploader("Upload your retail sales CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['date'] = pd.to_datetime(data['date'])
    data = data.groupby('date').sum().reset_index()
    prophet_data = data.rename(columns={"date": "ds", "sales": "y"})

    model = Prophet()
    model.fit(prophet_data)

    periods_input = st.slider("Select number of days to forecast", 30, 365)
    future = model.make_future_dataframe(periods=periods_input)
    forecast = model.predict(future)

    st.subheader("Forecasted Data")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    st.subheader("Forecast Plot")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

 #Run the Streamlit app using the command below:
 #streamlit run app.py
