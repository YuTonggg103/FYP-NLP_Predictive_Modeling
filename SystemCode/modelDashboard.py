import streamlit as st
from modelGraph import confusion_matrix ,PR_curve, learning_curve, ROC_curve

def get_color_class(score):
    score = float(score)
    if score >= 0.80:
        return "green"
    elif score >= 0.70:
        return "orange"
    else:
        return "red"
    
def run_model_dashboard(title,model):
    # CSS design
    st.markdown(f"""
        <style>
            .flip-card {{
                perspective: 1000px;
                height: 45px;
                margin-bottom: 15px;
            }}
            .flip-card-inner {{
                position: relative;
                width: 100%;
                height: 100%;
                text-align: center;
                transition: transform 0.6s;
                transform-style: preserve-3d;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            # .flip-card:hover .flip-card-inner {{
            #     transform: rotateY(180deg);
            # }}
            .flip-card-front, .flip-card-back {{
                position: absolute;
                width: 100%;
                height: 100%;
                backface-visibility: hidden;
                padding: 15px;
                border-radius: 10px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }}
            .flip-card-front {{
                background: rgba(255, 255, 255, 0.1);
                display: flex;
                align-items: center;
                justify-content: center;
                flex-direction: row;  
                gap: 10px;             
            }}
            .flip-card-back {{
                background: linear-gradient(#EEAECA,#94BBE9);
                # background: #102E50;
                color: black;
                transform: rotateY(180deg);
            }}
            .metric-title {{
                font-size: 20px;
                color: white;
            }}
            .metric-value {{
                font-size: 20px;
                font-weight: bold;
            }}
            .green {{
                color: #16C47F;  
            }}
            .orange {{
                color: #FF9800; 
            }}
            .red {{
                color: #E82561;  
            }}
            .back-text {{
                font-size: 16px;
                font-weight: bold;
            }}
            .nav-card {{
                background-color: rgba(255, 255, 255, 0.08);
                padding: 10px 5px;
                border: 1px solid rgba(200, 200, 200, 0.3);
                border-radius: 10px;
                text-align: center;
                font-weight: 600;
                font-size: 16px;
                transition: background-color 0.3s ease;
            }}
            .nav-card a {{
                text-decoration: none;
                color: #f0f0f0;
            }}
            .nav-card:hover {{
                background-color: rgba(255, 255, 255, 0.2);
            }}
        </style>
    """, unsafe_allow_html=True)

    if model == "GRU":
        metrics = {
            "Train Accuracy": {"value": "0.8283", "eval": ""},
            "Validation Accuracy": {"value": "0.7993", "eval": ""},
            "Precision": {"value": "0.6353", "eval": ""},
            "Recall": {"value": "0.8490", "eval": ""},
            "Weighted F1": {"value": "0.8054", "eval": ""},
            "Micro F1": {"value": "0.7993", "eval": ""},
            "Macro F1": {"value": "0.7841", "eval": ""}
        }
    elif model == "LSTM":
        metrics = {
            "Train Accuracy": {"value": "0.8371", "eval": ""},
            "Validation Accuracy": {"value": "0.7969", "eval": ""},
            "Precision": {"value": "0.6330", "eval": ""},
            "Recall": {"value": "0.8425", "eval": ""},
            "Weighted F1": {"value": "0.8030", "eval": ""},
            "Micro F1": {"value": "0.7969", "eval": ""},
            "Macro F1": {"value": "0.7813", "eval": ""}
        }
    elif model == "SenticNet":
        metrics = {
            "Train Accuracy": {"value": "0.8375", "eval": ""},
            "Validation Accuracy": {"value": "0.7871", "eval": ""},
            "Precision": {"value": "0.6116", "eval": ""},
            "Recall": {"value": "0.8846", "eval": ""},
            "Weighted F1": {"value": "0.7944", "eval": ""},
            "Micro F1": {"value": "0.7871", "eval": ""},
            "Macro F1": {"value": "0.7751", "eval": ""}
        }

    st.markdown(f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)
    st.divider()
    # Flip Card - Evaluation Matrix
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="nav-card"><a href="#confusion-matrix">📌 Confusion Matrix</a></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="nav-card"><a href="#model-metrics-overview">📊 Metrics</a></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="nav-card"><a href="#roc-auc-curve">🌊 ROC Curve</a></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="nav-card"><a href="#precision-recall-curve">🩴 PR Curve</a></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="nav-card"><a href="#learning-curve">📈 Learning Curve</a></div>', unsafe_allow_html=True)
    st.divider()

    with st.container():
        col1, col2 = st.columns(2)
        # Confusion Matrix
        with col1:
            st.markdown("<div id='confusion-matrix'></div>", unsafe_allow_html=True)
            st.subheader("Confusion Matrix")
            if model == "GRU":
                confusion_matrix(1169, 6572,13108, 3772, 22.35,15.10) 
            elif model == "LSTM":
                confusion_matrix(1219, 6522,13098, 3782, 22.41,15.75)
            elif model == "SenticNet":
                confusion_matrix(893, 6848,12531, 4349, 25.76,11.54)
        # Matrix
        with col2:
            st.markdown("<div id='model-metrics-overview'></div>", unsafe_allow_html=True)
            st.subheader("📊 Model Metrics Overview")
            with st.container():
                metric_names = ["Train Accuracy", "Validation Accuracy", "Precision", "Recall", "Weighted F1", "Micro F1", "Macro F1"]
                for metric in metric_names:
                    # st.markdown(f"""
                    #      <div class="flip-card">
                    #         <div class="flip-card-inner">
                    #             <div class="flip-card-front">
                    #                 <div class="metric-title">{metric}</div>
                    #                 <div class="metric-value {get_color_class(metrics[metric]['value'])}">{metrics[metric]['value']}</div>
                    #             </div>
                    #             <div class="flip-card-back">
                    #                 <div class="back-text">{metrics[metric]['eval']}</div>
                    #             </div>
                    #         </div>
                    #     </div>
                    # """, unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="flip-card">
                            <div class="flip-card-inner">
                                <div class="flip-card-front">
                                    <div class="metric-title">{metric}</div>
                                    <div class="metric-value {get_color_class(metrics[metric]['value'])}">{metrics[metric]['value']}</div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            st.divider()
        
        col1, col2 = st.columns(2)
        # ROC Curve
        with col1:
            st.markdown("<div id='roc-auc-curve'></div>", unsafe_allow_html=True)
            st.subheader("🌊 ROC-AUC Curve")
            if model == "GRU":
               ROC_curve(r"./json/Bi-GRU_ROC_Curve.json")
            elif model == "LSTM":
               ROC_curve(r"./json/Bi-LSTM_ROC_Curve.json")
            elif model == "SenticNet":
               ROC_curve(r"./json/Bi-LSTM_SenticNet_ROC_Curve.json")
        # PR Curve
        with col2:
            st.markdown("<div id='precision-recall-curve'></div>", unsafe_allow_html=True)
            st.subheader("🩴 Precision-Recall Curve")
            if model == "GRU":
               PR_curve(r"./json/Bi-GRU_PR_Curve.json")
            elif model == "LSTM":
               PR_curve(r"./json/Bi-LSTM_PR_Curve.json")
            elif model == "SenticNet":
               PR_curve(r"./json/Bi-LSTM_SenticNet_PR_Curve.json")
        # Learning Curve
        st.markdown("<div id='learning-curve'></div>", unsafe_allow_html=True)
        if model == "GRU":
            learning_curve(r"../Model Code/Saved Model Hate/GRU/6_GRU_Hate_FineTuning_History_1.pkl",3)
        elif model == "LSTM":
            learning_curve(r"../Model Code/Saved Model Hate/LSTM/FineTune_1_LSTM_FastText__History.pkl",8)
        elif model == "SenticNet":
            learning_curve(r"../Model Code/Saved Model Hate/EmoHate/4_BiLSTM_FastText_SenticNet_FocalLoss_History.pkl",7)
