import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import tensorflow.keras.backend as K
from tensorflow.keras.saving import register_keras_serializable
from nltk.tokenize import word_tokenize
from senticnet.senticnet import SenticNet
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import emoji
from tensorflow.keras.preprocessing.sequence import pad_sequences

def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'(http|https|www)\S+', '', text)
    html_tags_pattern = r'<.*?>'
    text = re.sub(html_tags_pattern, '',text)
    text = re.sub(r'\S+html\b', '',text)
    text = text = emoji.demojize(text)
    text = text = re.sub(r':\w+:', '', text)

    replace_dict = {
        '@': 'a',
        '$': 's',
        '$$':'ss',
        '0': 'o',
        '3': 'e',
        '1': 'i',
        '5': 's',
        '7': 't',
        '4': 'a',
        '9' : 'g',
    }
    def replace_symbols(text):
        for symbol, letter in replace_dict.items():
            text = re.sub(rf'{re.escape(symbol)}', letter, text)
            print(text) 
        return text
    text = replace_symbols(text)

    abbreviations = {
        "it's":"it is",
        "we're":"were are",
        "let's":"let us",
        "i'll":"i will",
    }
    def replace_abbreviations(text):
        text = str(text)
        for abbr, full_form in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)
        return str(text)
    text = replace_abbreviations(text)

    text = text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    text = text = re.sub(r'\s+', ' ', text).strip()
    text = text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    abbreviations = {
        "auser" : "",
        "werent":"were not","arent": "are not",
        "isnt": "is not",
        "cant": "can not",
        "shes": "she is","hes": "he is",
        "youre": "you are", 
        "youll": "you will",
        "youve": "you have",
        "weve": "we have",
        "yall":"you all",
        "theyre": "they are", 
        "theyve": "they have",
        "doesnt": "does not", 
        "dont":"do not",
        "didnt": "did not",
        "wont": "will not",
        "wouldnt": "would not",
        "shouldnt": "should not",
        "couldnt": "could not",
        "im": "i am",
        "iam": "i am",
        "ive": "i have",
        "id": "i would",
        "wth":"what the heal", "wtf":"what the fuck",
        "fk":"fuck", "f**k":"fuck","fu*k":"fuck", "f*ck":"fuck","fck":"fuck","fcking":"fucking","fking":"fucking",
        "cuz":"because", "bcuz":"because","becuz":"because",
        "bihday":"birthday",
        "etc":"et cetera",
        "selfie":"self portrait photograph",
        "lol":"laughing out loud",
        "lmao":"laughing my ass off",
        "forex":" foreign exchange",
        "lgbt":"transgender",
        "blm":"black lives matter",
        "obama":"Barack Obama",
        "omg":"oh my god",
        "ppl":"people",
        "fathersday":"father day",
    }
    def replace_abbreviations(text):
        text = str(text)
        for abbr, full_form in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)
        return str(text)
    text = replace_abbreviations(text)

    text = nltk.tokenize.word_tokenize(text)

    stopTokens = nltk.corpus.stopwords.words("english")
    stopTokens.remove('not') 
    stopTokens.remove('no') 
    def removeStopWord(words):
        return [word for word in words if word.lower() not in stopTokens]
    text = removeStopWord(text)

    def get_pos_tagging(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV} #need this for wordnet cuz wordnet only have 4 postag
        return tag_dict.get(tag, wordnet.NOUN)
    lemmatizer = WordNetLemmatizer()
    def lemmatize_text(text):
        lemmatized_words = [lemmatizer.lemmatize(word, get_pos_tagging(word)) for word in text] 
        return ' '.join(lemmatized_words)
    text = lemmatize_text(text)
    return text

# Predict Hate Function 
MAX_SEQUENCE_LENGTH = 100
NUM_SENTIC_FEATURES = 24 

@register_keras_serializable(package="CustomLoss")
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.55): 
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    cross_entropy = -K.log(pt)
    weight = alpha_t * K.pow((1 - pt), gamma)
    loss = weight * cross_entropy
    return K.mean(loss, axis=1)

@register_keras_serializable(package="CustomLoss")
def focal_loss_1(y_true, y_pred, gamma=2.0, alpha=0.55): 
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    cross_entropy = -K.log(pt)
    weight = alpha_t * K.pow((1 - pt), gamma)
    loss = weight * cross_entropy
    return K.mean(loss, axis=1)

# Load Model and Tokenizer
model = load_model(
    r"../Model Code/Saved Model Hate/EmoHate/4_BiLSTM_FastText_SenticNet_FocalLoss_Model.keras",
    custom_objects={'focal_loss_fixed': focal_loss}
)
model_GRU = load_model(
    r"../Model Code/Saved Model Hate/GRU/6_GRU_Hate_FineTuning_Model_1.keras",
    custom_objects={'focal_loss_1': focal_loss_1}
)
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Sentic Feature Extractor 
def extract_sentic_features(text, tokenizer, max_len):
    sn = SenticNet()
    sentic_matrix = np.zeros((max_len, NUM_SENTIC_FEATURES), dtype=np.float32)
    tokens = word_tokenize(text)
    token_indices = tokenizer.texts_to_sequences([tokens])[0]
    for i, token_idx in enumerate(token_indices[:max_len]):
        word = tokenizer.index_word.get(token_idx, '')
        try:
            sentic_values = [float(s) for s in sn.concept(word)[:NUM_SENTIC_FEATURES]]
            sentic_matrix[i] = sentic_values
        except:
            pass  # OOV or missing sentic entry → all zeros
    return sentic_matrix

# Prediction Function
def predict_hate(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, dtype="float32")

    sentic_input = np.array([extract_sentic_features(text, tokenizer, MAX_SEQUENCE_LENGTH)])

    # Model 1 prediction
    predictions = model.predict([padded_sequence, sentic_input])
    prob_1 = float(predictions.flatten()[0])
    hate_label = int(prob_1 > 0.5)
    result1 = "Hate Speech" if hate_label == 1 else "Non-Hate Speech"

    # Model 2 prediction
    predictions_2 = model_GRU.predict([padded_sequence, sentic_input])
    prob_2 = float(predictions_2.flatten()[0])
    hate_label_2 = int(prob_2 > 0.5)
    result2 = "Hate Speech" if hate_label_2 == 1 else "Non-Hate Speech"

    return result1, result2, prob_1, prob_2

