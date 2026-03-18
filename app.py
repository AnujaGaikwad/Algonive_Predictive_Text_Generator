
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load saved model and tokenizer
model = load_model("predictive_model.keras")
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# Retrieve the expected sequence length from the model’s input
# (We saved model.input_shape = (None, max_seq_len-1))
max_seq_len = model.input_shape[1] + 1

# 2. Prediction function with sampling and validation
def predict_next_words(text_input, num_words=1, temperature=1.0):
    text_input = text_input.lower()
    words = text_input.split()
    if not words:
        return ""  # No input text
    for _ in range(num_words):
        # Convert current words to padded sequence
        token_list = tokenizer.texts_to_sequences([' '.join(words)])[0]
        if not token_list:
            break  # No valid tokens
        padded = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        preds = model.predict(padded, verbose=0)[0]

        # Apply temperature for more diverse sampling (optional)
        if temperature != 1.0:
            preds = np.log(preds) / temperature
            preds = np.exp(preds) / np.sum(np.exp(preds))

        # Sample the next word index from the probability distribution
        next_index = np.random.choice(range(len(preds)), p=preds)
        next_word = tokenizer.index_word.get(next_index, "")
        if not next_word:
            break  # Safety check
        words.append(next_word)
    # Return only the newly generated words
    return ' '.join(words[-num_words:])

# 3. Streamlit UI
st.title("🤖 AI Text Predictor")
st.write("Type a prompt below and the model will predict the next word(s).")
user_input = st.text_input("Enter text:")

if user_input:
    result = predict_next_words(user_input, num_words=1, temperature=1.0)
    st.success(f"▶ Next word: **{result}**")
