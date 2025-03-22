import pandas as pd
import json
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence
from torchsummary import summary
import torch.nn as nn


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
# Function to map tokens to vectors

def tokens_to_vectors(tokens):
    return [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]

df["post_vectors"] = df["post_tokens"].apply(tokens_to_vectors)

#print(df["post_vectors"].iloc[0])
#print(f"Vector shape for first word: {df['post_vectors'].iloc[0][0].shape}")

# First split: Train 70% | Temp 30%
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["post_label"])

# Second split: Validation 15% | Test 15% from Temp
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["post_label"])

# Confirm the sizes
#print(f"Train size: {len(train_df)}")
#print(f"Validation size: {len(val_df)}")
#print(f"Test size: {len(test_df)}")


label2id = {"normal": 0, "hatespeech": 1, "offensive": 2}
df["label_id"] = df["post_label"].map(label2id)


# Convert list of vectors to padded tensors and labels
def prepare_split(df_split):
    # Filter out posts with empty vectors
    df_filtered = df_split[df_split["post_vectors"].apply(lambda v: len(v) > 0)].copy()

    # Convert labels
    y = torch.tensor(df_filtered["post_label"].map(label2id).values, dtype=torch.long)

    # Convert each list of vectors to a tensor
    X = [torch.tensor(vectors, dtype=torch.float) for vectors in df_filtered["post_vectors"]]

    # Pad sequences
    X_padded = pad_sequence(X, batch_first=True)
    #print(f"Removed {len(df_split) - len(df_filtered)} posts with empty vectors.")

    return X_padded, y



# Prepare data for each split
X_train, y_train = prepare_split(train_df)
X_val, y_val     = prepare_split(val_df)
X_test, y_test   = prepare_split(test_df)

# Check shapes
#print("Train:", X_train.shape, y_train.shape)
#print("Val:", X_val.shape, y_val.shape)
#print("Test:", X_test.shape, y_test.shape)



# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()  
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # hidden = [num_layers, batch_size, hidden_dim]
        out = self.fc(hidden[-1])      # Get the output from the last LSTM layer
        return out

# Define model parameters
input_dim = 100      # Word2Vec vector size
hidden_dim = 128
output_dim = 3       # Number of classes

# Instantiate the model
model = LSTMClassifier(input_dim, hidden_dim, output_dim)
print(model)

