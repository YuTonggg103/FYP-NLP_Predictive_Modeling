import streamlit as st
import base64
import random
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
from datasetDashboard import run_dataset_dashboard 
from modelDashboard import run_model_dashboard 
from predictHate import text_preprocessing,predict_hate

st.set_page_config(layout="wide")

# ---------------------------------------------------------------------------------------------------------------------

# Backgroud Image
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
image_path = r"./image/blue-background-7470781_1280.jpg"
image_base64 = image_to_base64(image_path)

# CSS Design
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('data:image/png;base64,{image_base64}');
        background-size: cover;
        background-position: center;
    }}
    [data-testid=stSidebar] {{
        background-color: #1e1f25;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
.stTextArea textarea {
    background-color: rgba(8, 3, 6, 0.25);
    color: white;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 10px;
    font-size: 16px;
}
.result-container {
    margin-top: 15px;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #6e8efb;
    background-color: rgba(0, 0, 0, 0.3);
    color: white;
}
.hate-border {
    border: 2px solid #123458;
}
.nonhate-border {
    border: 2px solid #123458;
}
.progress-bar {
    background-color: #e0e0e0;
    border-radius: 5px;
    height: 20px;
    margin-top: 10px;
    overflow: hidden;
}
.progress {
    height: 100%;
    text-align: center;
    color: white;
    font-weight: bold;
    line-height: 20px;
}
.progress-hate {
    background: linear-gradient(90deg, #ff4d4f, #ff7875);
}
.progress-nonhate {
    background: linear-gradient(90deg, #52c41a, #73d13d);
}
</style>
""", unsafe_allow_html=True)

# Web Page Content----------------------------------------------------------------------------------------------------------------
@st.cache_data
def load_sample_data():
    df = pd.read_csv("Model Deploy Sample Data.csv")
    data = df[df['label'] == 1]['text'].tolist()
    return data

sample_texts = load_sample_data()

st.sidebar.title("Select a Page")
main_page = st.sidebar.selectbox("", [
    "Hate Speech Predict Page",
    "Hate Speech Dataset Dashboard",
    "Model Preformances Dashboard"
])

if main_page == "Hate Speech Predict Page":
    st.markdown(""" <h1 style="text-align: center; font-size: 36px;">Hate Speech Detection System</h1>
                <hr style="height:2px; border:none; background-color:#ccc;" /> """, unsafe_allow_html=True)
    # Initialize session states
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if 'history' not in st.session_state:
        st.session_state.history = []

    st.markdown("""<h3 style=" font-family: 'Arial', sans-serif; font-size: 24px; color: white; font-weight: 600; margin-bottom: 10px;
    ">📝 Please Enter Your Text: </h3> """, unsafe_allow_html=True)
    user_input = st.text_area("", value=st.session_state.user_input)

    col1, col2 = st.columns([1, 9])
    with col1:
        check = st.button("  Check 🚨")
    with col2:
        if st.button("  Sample Hate Text 🎲 "):
            if sample_texts:
                st.session_state.user_input = random.choice(sample_texts)
                st.rerun()
            else:
                st.warning("No hate speech samples found in the dataset.")
    if check:
        if user_input.strip(): 
            try:
                # After Text preprocessing
                processed_text = text_preprocessing(user_input)
                st.markdown("""<div style=" background-color: rgba(0, 0, 0, 0.5); padding: 15px; border-radius: 12px; margin-bottom: 20px;"> 
                                    <h3 style="margin-top: 0; font-size: 20px;font-family: 'Arial', sans-serif;">After Text Preprocessing: </h3>
                                    <p style="color: white;">{}</p>
                                </div> """.format(processed_text), unsafe_allow_html=True)

                result_1, result_2, prob_1, prob_2 = predict_hate(processed_text)

                # Add to history
                history_entry = {
                    "sentence": user_input,
                    "BiLSTM-SenticNet": f"{result_1}",
                    "Prob_1": f"{round(prob_1*100, 2)}%",
                    "Bi-GRU": f"{result_2}",
                    "Prob_2": f"{round(prob_2*100, 2)}%",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                st.session_state.history.insert(0, history_entry)
                if len(st.session_state.history) > 10:
                    st.session_state.history = st.session_state.history[:10]

                # Display prediction
                if result_1 == "Hate Speech":
                    border_class_1 = "hate-border"
                else:
                    border_class_1 = "nonhate-border"
                if result_2 == "Hate Speech":
                    border_class_2 = "hate-border"
                else:
                    border_class_2 = "nonhate-border"

                percent_1 = round(prob_1 * 100, 2)
                percent_2 = round(prob_2 * 100, 2)
                color_class_1 = "progress-hate" if result_1 == "Hate Speech" else "progress-nonhate"
                color_class_2 = "progress-hate" if result_2 == "Hate Speech" else "progress-nonhate"

                st.markdown(f"""
                <div class="result-container {border_class_1}" style="padding: 15px; border-radius: 10px; margin-bottom: 20px; background-color: rgba(0, 0, 0, 0.5);">
                    <p style="margin: 0; font-size: 18px;">
                        ▶️ <b>BiLSTM + SenticNet Prediction:</b> <span style="color:white;">{result_1}</span> 
                        <span style="float:right;"><b>{percent_1}%</b></span>
                    </p>
                    <div class="progress-bar" style="margin-top: 8px;">
                        <div class="progress {color_class_1}" style="width: {percent_1}%; height: 20px;">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="result-container {border_class_2}" style="padding: 15px; border-radius: 10px; margin-bottom: 30px; background-color: rgba(0, 0, 0, 0.5);">
                    <p style="margin: 0; font-size: 18px;">
                        ▶️ <b>Bi-GRU Prediction:</b> <span style="color:white;">{result_2}</span> 
                        <span style="float:right;"><b>{percent_2}%</b></span>
                    </p>
                    <div class="progress-bar" style="margin-top: 8px;">
                        <div class="progress {color_class_2}" style="width: {percent_2}%; height: 20px;">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"Error 1: {e}")
        else:
            st.warning("Please enter text 😈")
    
    # History Table 
    st.divider()
    st.markdown("""
    <h3 style=" font-family: 'Arial', sans-serif; font-size: 24px; color: white; font-weight: 600; margin-bottom: 10px; ">📄Prediction History (Last 10):</h3>
    """, unsafe_allow_html=True)
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        history_df.index = np.arange(1, len(history_df) + 1 )
        
        st.dataframe(
            history_df[["sentence", "BiLSTM-SenticNet", "Prob_1","Bi-GRU","Prob_2"]],
            column_config={
                "sentence": "Input Text",
                "BiLSTM-SenticNet": "BiLSTM-SenticNet",
                "Prob_1": "Probability",
                "Bi-GRU": "Bi-GRU",
                "Prob_2": "Probability",
            },
            use_container_width=True,
            hide_index=False
        )
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No prediction history yet.")

elif main_page == "Hate Speech Dataset Dashboard":
    run_dataset_dashboard()

elif main_page == "Model Preformances Dashboard":
    model_option = st.sidebar.radio("Select a Model", [
        "Bi-GRU", "BiLSTM-SenticNet", "Bi-LSTM" ])
    if model_option == "Bi-GRU":
        run_model_dashboard("Bi-GRU Hate Base Deep Learning Model","GRU")
    elif model_option == "BiLSTM-SenticNet":
        run_model_dashboard("BiLSTM-SenticNet Concept-Level Fusion Model ","SenticNet")
    elif model_option == "Bi-LSTM":
        run_model_dashboard("Bi-LSTM Hate Base Deep Learning Model","LSTM")



