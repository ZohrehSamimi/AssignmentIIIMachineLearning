import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ast
import gensim
print("Gensim installed successfully! Version:", gensim.__version__)
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Load cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

# Ensure tokens are stored as lists
df["post_tokens"] = df["post_tokens"].apply(ast.literal_eval)

print("Dataset loaded successfully!")

# Load the Word2Vec model
word2vec = KeyedVectors.load("Data/word2vec.model")

print("Word2Vec model loaded!")
print("Vocabulary size:", len(word2vec.index_to_key))

# Convert tokens to Word2Vec vectors
VECTOR_SIZE = 300  # Adjust based on your Word2Vec model

def tokens_to_vectors(tokens, word2vec, vector_size=VECTOR_SIZE):
    """
    Convert a list of tokens into Word2Vec vectors.
    Words not found in Word2Vec will be replaced with zero vectors.
    """
    vectors = [word2vec[token] if token in word2vec else np.zeros(vector_size) for token in tokens]
    return np.array(vectors)

# Apply function
df["vector_representation"] = df["post_tokens"].apply(lambda tokens: tokens_to_vectors(tokens, word2vec))

print("Token conversion complete!")

# Pad sequences
max_length = df["post_tokens"].apply(len).max()
print(f"Maximum sequence length: {max_length}")

padded_vectors = pad_sequences(df["vector_representation"].tolist(), maxlen=max_length, dtype="float32", padding="post", truncating="post")

print("Padding complete! Shape:", padded_vectors.shape)

# Save processed data
np.save("word2vec_padded_vectors.npy", padded_vectors)
df["post_label"].to_csv("labels.csv", index=False)

print("Preprocessed data saved successfully!")
