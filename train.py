
import re
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 1. Load and clean data
with open("data/corpus.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()
text = re.sub(r'[^\w\s]', '', text)  # remove punctuation

# 2. Tokenize and create sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[: i+1])

max_seq_len = max(len(seq) for seq in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# 3. Build and train the model
model = Sequential([
    Embedding(total_words, 50, input_length=max_seq_len-1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=50, verbose=1)

# 4. Save the model and tokenizer
model.save("predictive_model.keras")  # Save entire model (architecture + weights)【12†L92-L100】
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)  # Save tokenizer object【5†L150-L158】
