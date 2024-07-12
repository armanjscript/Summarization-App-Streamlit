import streamlit as st
from model import mT5


@st.cache_resource(show_spinner="Model is getting prepared...")
def get_model():
    model = mT5()
    return model

st.header('Summarization App 📱', divider='rainbow')

with st.form('summarize_app'):
    input_text = st.text_area("Your Text ...")
    submit_btn = st.form_submit_button("Summarize My Text!")

if input_text!="":
    model = get_model()
    output_text=model.run(input_text)
    st.write(f"Your Text Summarization: {output_text}")
    
    