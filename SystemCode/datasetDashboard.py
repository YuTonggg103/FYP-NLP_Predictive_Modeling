import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import words, wordnet
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from collections import Counter

def run_dataset_dashboard():
    # nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('words')

    file_path = r"..\Pre_Hate_Dataset\20_Hate_Final.csv"
    data = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

    with st.container():
        st.markdown("""
            <h1 style="text-align: center; font-size: 36px;">Hate Speech Dataset Dashboard</h1>
            <hr style="height:2px; border:none; background-color:#ccc;" />
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            # 1 Hate & Non-Hate Distribution
            st.subheader("📊 Hate & Non-Hate Distribution") 
            def hate_NoHate_percenatge(label):
                if label == 1:
                    return "Hate"
                elif label == 0:
                    return "Non-Hate"
                return "Unknown"
            data['label'] = data['label'].apply(hate_NoHate_percenatge)
            label_counts = data['label'].value_counts().reset_index()
            label_counts.columns = ['Label', 'Count']
            
            fig_pie = px.pie(label_counts, names='Label', values='Count',
                            color='Label', color_discrete_map={'Hate': '#211C84', 'Non-Hate': '#7A73D1'},
                            hole=0.3)
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0.5)',   # chart transparent background 
                legend=dict(
                    font=dict(size=18, family='Arial',),
                    orientation="h",  
                    x=0.5,            # legend center
                    xanchor='center'  # pie chart center
                )
            )
            fig_pie.update_traces(textinfo='percent+label')
            fig_pie.update_traces(
                textinfo='percent+label',
                textfont=dict(size=18, family='Arial')
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # 2. Enhanced Word Cloud
            st.subheader("☁️ Word Cloud (Hate only)", divider="blue")
            # Filter only hate speech for word cloud
            hate_text = ' '.join(data[data['label'] == 'Hate']['text'].astype(str))
            words_1 = word_tokenize(hate_text)
            word_counts = Counter(words_1)
            
            wordcloud = WordCloud(
                width=800,
                height=500,
                background_color=None,
                mode='RGBA',
                max_words=150,
                colormap='Set3',
                prefer_horizontal=0.9,
                min_font_size=8,
                max_font_size=150,
                relative_scaling=0.5,
                font_path=None,
                random_state=42,
                collocations=False,
            ).generate_from_frequencies(word_counts)
            
            fig_wc, ax_wc = plt.subplots(figsize=(12, 8), facecolor='none') #facecolor -> Background Transparent
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            plt.tight_layout(pad=0)
            fig_wc.patch.set_alpha(0)
            
            st.pyplot(fig_wc, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
           # 3 Bigrams
            st.subheader("🔗 Most Frequency Bigrams") 
            label_option = st.selectbox( "Choose the label to show bigrams:", options=["Hate", "Non-Hate"])

            data_label_1 = data[data['label'] == label_option]
            vectorizer = CountVectorizer(ngram_range=(2, 2))
            X = vectorizer.fit_transform(data_label_1['text'].astype(str))
            bigram_counts = np.array(X.sum(axis=0)).flatten()
            bigram_freq = pd.DataFrame({
                'bigram': vectorizer.get_feature_names_out(),
                'frequency': bigram_counts
            }).sort_values(by='frequency', ascending=False).head(20)

            fig_bigram = go.Figure()
            fig_bigram.add_trace(go.Bar(
                x=bigram_freq['frequency'],
                y=bigram_freq['bigram'],
                orientation='h',
                marker=dict(color='#48A6A7', line=dict(color='rgba(0,0,0,0.1)', width=1)),
                text=bigram_freq['frequency'],
                textposition='auto',
                textfont=dict(family="Arial", size=14, color="white", weight="bold")  # Adding font style here
            ))

            fig_bigram.update_layout(
                xaxis_title='Frequency',
                yaxis_title='Bigram',
                yaxis=dict(autorange="reversed"),  # Most Frequent on top
                paper_bgcolor='rgba(0,0,0,0.5)',
                font=dict(size=14, family='Arial'),
                margin=dict(t=60, l=80, r=40, b=50),
                height=700
            )
            st.plotly_chart(fig_bigram, use_container_width=True)

        with col2:
            # 4 Unknown Word
            st.subheader("🔍 Top 20 Unknown Words") 
            label_option = st.selectbox( "Choose a label:", options=["Before Text Preprocessing", "After Text Preprocessing"])

            if label_option == "Before Text Preprocessing":
                file_path = r"..\Pre_Hate_Dataset\ForDashBoardUsed.csv"
                data = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

            combined_vocab = set(words.words()) | set(wordnet.words())
            def find_unknown_words_combined(text):
                if not isinstance(text, str): return []
                words_in_text = re.findall(r'\b\w+\b', text.lower())
                return [word for word in words_in_text if word not in combined_vocab]

            data['unknown_words'] = data['text'].apply(find_unknown_words_combined)
            all_unknown_words = [word for word_list in data['unknown_words'] for word in word_list]
            unknown_word_freq = Counter(all_unknown_words)
            freq_df = pd.DataFrame(unknown_word_freq.items(), columns=['word', 'frequency']).sort_values(by='frequency', ascending=False)
            top_unknown = freq_df.head(20)

            # Horizontal bar chart
            fig_unknown = go.Figure()
            fig_unknown.add_trace(go.Bar(
                x=top_unknown['frequency'],
                y=top_unknown['word'],
                orientation='h',
                marker=dict(color='#A97DCE', line=dict(color='rgba(0,0,0,0.1)', width=1)),
                text=top_unknown['frequency'],
                textposition='auto',
                textfont=dict(family="Arial", size=14, color="white")
            ))

            fig_unknown.update_layout(
                xaxis_title='Frequency',
                yaxis_title='Word',
                yaxis=dict(autorange="reversed"),  
                paper_bgcolor='rgba(0,0,0,0.5)',  
                font=dict(size=14, family='Arial'),
                margin=dict(t=60, l=80, r=40, b=50),
                height=700
            )
            st.plotly_chart(fig_unknown, use_container_width=True)