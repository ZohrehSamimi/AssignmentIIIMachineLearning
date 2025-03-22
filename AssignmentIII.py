import pandas as pd
import json
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string
from gensim.models import Word2Vec
import numpy as np


# Download stopwords if not already present
nltk.download('stopwords')

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Load dataset.json
with open('dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Convert to DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

'''What the Code Does
Extracts labels from the "annotators" column.
Counts the frequency of each label for a post.
Assigns the most frequent label as the "post_label".'''
# Function to extract majority label
def get_majority_label(annotators):
    labels = [ann["label"] for ann in annotators]  # Extract labels
    most_common_label = Counter(labels).most_common(1)[0][0]  # Find the most frequent label
    return most_common_label

# Apply function to create post_label column
df["post_label"] = df["annotators"].apply(get_majority_label)
print(df["post_label"].value_counts())  # Check class distribution

# Lowercase all tokens in post_tokens
df["post_tokens"] = df["post_tokens"].apply(lambda tokens: [token.lower() for token in tokens])

#print(df["post_tokens"].head())

# Remove stopwords from post_tokens
df["post_tokens"] = df["post_tokens"].apply(lambda tokens: [token for token in tokens if token not in stop_words])
#print(df["post_tokens"].head())

# Remove punctuation-only tokens
df["post_tokens"] = df["post_tokens"].apply(
    lambda tokens: [token for token in tokens if token not in string.punctuation and not all(char in string.punctuation for char in token)]
)
#print(df["post_tokens"].head())



# Display updated DataFrame
print(df[["post_id", "post_label", "post_tokens"]].head())


#This way, you won’t need to process dataset.json again!
df.to_csv("cleaned_dataset.csv", index=False)
print("Dataset saved as cleaned_dataset.csv")

#Train Word2Vec model on your tokenized data
# Train Word2Vec model
w2v_model = Word2Vec(sentences=df["post_tokens"], vector_size=100, window=5, min_count=2, workers=4, sg=1)
#check the Model
#print(w2v_model.wv["burger"])  # Example: See vector for a word
#print(w2v_model.wv.most_similar("burger"))  # Similar words



