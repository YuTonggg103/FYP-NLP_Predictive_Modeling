# FYP


**Topic:** Emotion-Aware Hate Speech Detection in User-Generated Content Using Natural Language Processing

**PDF:** https://github.com/YuTonggg103/FYP-NLP_Predictive_Modeling/blob/main/PDF_FYP%20Documentation.pdf

This project is to detect hate speech by incorporating emotion features into the natural language processing (NLP) model. Enhanced text preprocessing step to overcome traditional keyword-based model. By comparing the traditional machine learning model, deep learning model and integrate emotion feature model, to test whether the integration of emotion features can help the model improve the performance of detecting hate speech. 

**Model Used:** Bi-LSTM, Bi-GRU, CNN-BiLSTM, BiLSTM Multi-task model, Bi-LSTM Concept-Level Fusion Model, XGBoost.

**Technical Used:** FastText, Word2Vec, Focal Loss, ReduceLROnPlateau, Early Stopping, RandomSearch

**Hate Speech Dataset Sources:**
1. https://github.com/sharmaroshan/Twitter-Sentiment-Analysis
2. https://github.com/Vicomtech/hate-speech-dataset/tree/master
3. https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset/tree/main
4. https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/tree/master/ethos/ethos_data 

**Emotion Dataset:**
1. https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset/data

**Emotion Concept Level Fusion:**
1. imported from the “SenticNet” Python library

**Model Deployment:**
1. Streamlit Library

