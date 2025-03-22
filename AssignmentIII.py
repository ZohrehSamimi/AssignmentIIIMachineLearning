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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score

# Download stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
with open('dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
df = pd.DataFrame.from_dict(data, orient='index')

# Extract majority labels
def get_majority_label(annotators):
    labels = [ann["label"] for ann in annotators]
    return Counter(labels).most_common(1)[0][0]
df["post_label"] = df["annotators"].apply(get_majority_label)
print(df["post_label"].value_counts())

# Preprocessing
df["post_tokens"] = df["post_tokens"].apply(lambda tokens: [token.lower() for token in tokens])
df["post_tokens"] = df["post_tokens"].apply(lambda tokens: [t for t in tokens if t not in stop_words])
df["post_tokens"] = df["post_tokens"].apply(lambda tokens: [t for t in tokens if t not in string.punctuation and not all(c in string.punctuation for c in t)])
print(df[["post_id", "post_label", "post_tokens"]].head())

# Save cleaned version
df.to_csv("cleaned_dataset.csv", index=False)
print("Dataset saved as cleaned_dataset.csv")

# Train Word2Vec
w2v_model = Word2Vec(sentences=df["post_tokens"], vector_size=100, window=5, min_count=2, workers=4, sg=1)
df["post_vectors"] = df["post_tokens"].apply(lambda tokens: [w2v_model.wv[token] for token in tokens if token in w2v_model.wv])

# Split dataset
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["post_label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["post_label"])
label2id = {"normal": 0, "hatespeech": 1, "offensive": 2}

def prepare_split(df_split):
    df_filtered = df_split[df_split["post_vectors"].apply(lambda v: len(v) > 0)].copy()
    y = torch.tensor(df_filtered["post_label"].map(label2id).values, dtype=torch.long)
    X = [torch.tensor(vectors, dtype=torch.float) for vectors in df_filtered["post_vectors"]]
    X_padded = pad_sequence(X, batch_first=True)
    return X_padded, y

X_train, y_train = prepare_split(train_df)
X_val, y_val = prepare_split(val_df)
X_test, y_test = prepare_split(test_df)

# Model definition
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(self.dropout(hidden))

# === Baseline Model Training ===
hidden_dim = 128
dropout = 0.5
learning_rate = 1e-3
batch_size = 64
num_epochs = 20

model = LSTMClassifier(input_dim=100, hidden_dim=hidden_dim, output_dim=3,
                       num_layers=1, dropout=dropout, bidirectional=True).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    train_acc = 100 * correct / total
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == batch_y).sum().item()
            val_total += batch_y.size(0)
    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# Baseline Evaluation
model.eval()
val_preds, val_labels = [], []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        val_preds.extend(predicted.cpu().numpy())
        val_labels.extend(batch_y.cpu().numpy())

report_dict = classification_report(val_labels, val_preds, target_names=["normal", "hatespeech", "offensive"], output_dict=True)
val_acc = accuracy_score(val_labels, val_preds)

baseline_results = pd.DataFrame({
    "model": ["BiLSTM"],
    "hidden_dim": [hidden_dim],
    "dropout": [dropout],
    "lr": [learning_rate],
    "val_accuracy": [val_acc],
    "normal_f1": [report_dict["normal"]["f1-score"]],
    "hatespeech_f1": [report_dict["hatespeech"]["f1-score"]],
    "offensive_f1": [report_dict["offensive"]["f1-score"]],
    "macro_f1": [report_dict["macro avg"]["f1-score"]]
})
baseline_results.to_csv("model_results.csv", index=False)
print("✅ Baseline evaluation saved to model_results.csv")

# === Hyperparameter Tuning ===
hidden_dims = [64, 128]
dropouts = [0.3, 0.5]
learning_rates = [1e-3, 5e-4]
batch_sizes = [32, 64]

experiments = []
best_model = None
best_score = 0

for hidden_dim in hidden_dims:
    for dropout in dropouts:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                print(f"\n🔧 Training with hidden_dim={hidden_dim}, dropout={dropout}, lr={lr}, batch={batch_size}")
                train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

                model = LSTMClassifier(input_dim=100, hidden_dim=hidden_dim, output_dim=3,
                                       num_layers=1, dropout=dropout, bidirectional=True).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                loss_fn = nn.CrossEntropyLoss()

                for epoch in range(10):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = loss_fn(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        _, predicted = torch.max(outputs, 1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(batch_y.cpu().numpy())

                report = classification_report(all_labels, all_preds, target_names=["normal", "hatespeech", "offensive"], output_dict=True)
                macro_f1 = report["macro avg"]["f1-score"]
                acc = accuracy_score(all_labels, all_preds)

                experiments.append({
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "lr": lr,
                    "batch_size": batch_size,
                    "accuracy": acc,
                    "macro_f1": macro_f1
                })

                if macro_f1 > best_score:
                    best_score = macro_f1
                    best_model = model

                print(f"✅ Val Acc: {acc:.4f} | Macro F1: {macro_f1:.4f}")

# Save tuning results
tuned_df = pd.DataFrame(experiments)
tuned_df.to_csv("tuned_model_results.csv", index=False)
print("📊 Tuning results saved to tuned_model_results.csv")

# Print best configuration
best_exp = max(experiments, key=lambda x: x["macro_f1"])
print("\n🏆 Best Hyperparameters:")
for key, val in best_exp.items():
    print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")
