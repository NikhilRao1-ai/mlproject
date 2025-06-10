import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the tokenizer
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Test tokenization
text = "We are hiring a Software Engineer to work on machine learning projects. The ideal candidate should have experience with Python, Flask, and TensorFlow. Knowledge of CNN models is a plus."
sequences = tokenizer.texts_to_sequences([text])
print("Tokenized sequences:", sequences)

# Print the vocabulary
print("Vocabulary:", tokenizer.word_index)