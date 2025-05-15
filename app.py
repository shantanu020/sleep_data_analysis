# app.py

import streamlit as st
import pickle
import numpy as np

# Load saved model
model = pickle.load(open('model.pkl', 'rb'))

# Title
st.title('🛌 Sleep Health - Stress Prediction App')

st.write("""
This app predicts whether your current sleep and lifestyle indicates **High Stress** or **Low Stress**.
Fill in your details and click Predict! 🚀
""")

# Input fields
sleep_duration = st.number_input('🛏️ Sleep Duration (hours)', min_value=0.0, max_value=24.0, step=0.1)
quality_of_sleep = st.slider('🌟 Quality of Sleep (1 worst - 10 best)', 1, 10)
physical_activity = st.slider('🏃 Physical Activity Level (0-100)', 0, 100)
heart_rate = st.number_input('❤️ Heart Rate (beats per minute)', min_value=50, max_value=120)
daily_steps = st.number_input('👟 Daily Steps Walked', min_value=0, max_value=20000)

# Predict
if st.button('🔮 Predict Stress Level'):
    input_data = np.array([[sleep_duration, quality_of_sleep, physical_activity, heart_rate, daily_steps]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error('⚠️ High Stress Detected! Take Care 🧘‍♂️')
    else:
        st.success('✅ Low Stress! Keep it up! 🌟')

# Footer
st.markdown("---")
st.caption('Made with ❤️ using Streamlit')
