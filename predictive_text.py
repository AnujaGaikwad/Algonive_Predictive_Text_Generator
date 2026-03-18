import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ========================
# 1. LOAD DATASET
# ========================
with open("data/corpus.txt", "r", encoding="utf-8") as file:
    text = file.read()

# ========================
# 2. CLEAN TEXT
# ========================
text = text.lower()
text = re.sub(r'[^\w\s]', '', text)

# ========================
# 3. TOKENIZER
# ========================
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

# ========================
# 4. CREATE SEQUENCES
# ========================
input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_sequences.append(n_gram_seq)

# ========================
# 5. PAD SEQUENCES
# ========================
max_seq_len = max(len(seq) for seq in input_sequences)

input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
)

# ========================
# 6. SPLIT X AND y
# ========================
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# ========================
# 7. BUILD MODEL
# ========================
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_seq_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ========================
# 8. TRAIN MODEL
# ========================
print("Training model...")
model.fit(X, y, epochs=50, verbose=1)

# ========================
# 9. PREDICT FUNCTION
# ========================
def predict_next_word(text_input):
    token_list = tokenizer.texts_to_sequences([text_input])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

    predicted = np.argmax(model.predict(token_list), axis=-1)

    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word

# ========================
# 10. TEST
# ========================
print("\nPrediction:")
print("Input: machine learning")
print("Next word:", predict_next_word("machine learning"))