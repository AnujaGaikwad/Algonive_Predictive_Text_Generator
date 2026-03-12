import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide tensorflow logs

import re
from tensorflow.keras.preprocessing.text import Tokenizer

# read dataset
with open("data/corpus.txt", "r", encoding="utf-8") as file:
    text = file.read()

# cleaning
text = text.lower()
text = re.sub(r'[^\w\s]', '', text)

print("Dataset Loaded:\n", text)

# create tokenizer
tokenizer = Tokenizer()

# fit tokenizer
tokenizer.fit_on_texts([text])

# word index
word_index = tokenizer.word_index

print("\nWord Index:")
print(word_index)

print("\nVocabulary Size:", len(word_index))