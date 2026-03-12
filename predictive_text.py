import re

# read dataset
with open("data/corpus.txt", "r", encoding="utf-8") as file:
    text = file.read()

# convert to lowercase
text = text.lower()

# remove punctuation
text = re.sub(r'[^\w\s]', '', text)

# tokenize
words = text.split()

print("Total words:", len(words))
print("Sample words:", words[:20])