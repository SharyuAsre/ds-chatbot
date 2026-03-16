import nltk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random
import os

# --------------------------------------------------
# NLTK downloads (FIX for punkt_tab error)
# --------------------------------------------------
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --------------------------------------------------
# Load model and required files safely
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'ds_chatbot_model.h5'))

with open(os.path.join(BASE_DIR, 'chatbot_data.json'), encoding='utf-8') as f:
    intents = json.load(f)

with open(os.path.join(BASE_DIR, 'words.pkl'), 'rb') as f:
    words = pickle.load(f)

with open(os.path.join(BASE_DIR, 'classes.pkl'), 'rb') as f:
    classes = pickle.load(f)

lemmatizer = WordNetLemmatizer()

# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# --------------------------------------------------
# Bag of Words
# --------------------------------------------------
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# --------------------------------------------------
# Predict intent
# --------------------------------------------------
def predict_class(sentence, model, error_threshold=0.25):
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]

    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)

    return [{"intent": classes[r[0]], "probability": float(r[1])} for r in results]

# --------------------------------------------------
# Get response
# --------------------------------------------------
def getResponse(ints, intents_json):
    if not ints:
        return "Sorry, I didn’t understand that. Please rephrase your question."

    tag = ints[0]['intent']
    for intent in intents_json.get('intents', []):
        if intent.get('tag') == tag:
            return random.choice(intent.get('responses', []))

    return "I’m not sure how to respond to that."

# --------------------------------------------------
# Main chatbot function
# --------------------------------------------------
def chatbot_response(msg):
    try:
        ints = predict_class(msg, model)
        return getResponse(ints, intents)
    except Exception as e:
        print("Chatbot error:", e)
        return "Something went wrong on the server side. Please try again."