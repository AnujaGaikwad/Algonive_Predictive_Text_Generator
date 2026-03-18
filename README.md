Predictive Text Generator (LSTM NLP Project)
📌 Overview

This project is an AI-based Predictive Text Generator built using Natural Language Processing (NLP) and Deep Learning (LSTM).

It predicts the next word in a sentence based on previously typed text — similar to autocomplete systems used in keyboards and chat applications.

🎯 Objective

To design and implement a sequence-based NLP model that learns language patterns and generates context-aware next-word predictions.

🛠 Tech Stack

Python

TensorFlow / Keras

NumPy

Streamlit (for UI)

Regex (Text preprocessing)

⚙️ Features

🔮 Next word prediction

🧠 LSTM-based sequence learning

📝 Text preprocessing & tokenization

⚡ Fast prediction using saved model

🌐 Interactive web app using Streamlit

🧠 How It Works

Load and clean text dataset

Convert text into tokens

Generate n-gram sequences

Train LSTM model on sequences

Save trained model & tokenizer

Predict next word using trained model

▶️ Example

Input:

machine learning

Output:

machine learning is in ai
📂 Project Structure
Algonive_Predictive_Text_Generator
│
├── data/
│   └── corpus.txt
├── train.py
├── app.py
├── requirements.txt
└── README.md
🚀 How to Run
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Train the model
python train.py
3️⃣ Run the web app
streamlit run app.py
💡 Future Improvements

Use larger dataset for better accuracy

Add multi-word prediction

Improve sentence generation using temperature sampling

Deploy on cloud (Streamlit Cloud / Render)

👩‍💻 Author

Anuja Ramesh Gaikwad
