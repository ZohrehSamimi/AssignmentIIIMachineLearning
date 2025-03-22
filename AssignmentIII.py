# --- Imports ---
import pandas as pd, json, string, nltk, torch, numpy as np
from collections import Counter
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# --- Settings ---
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CONFIG FLAGS ===
DO_TRAIN = True
DO_TUNING = True

# === STEP 1: Load and Preprocess Dataset ===
def load_and_preprocess():
    with open("dataset.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    df = pd.DataFrame.from_dict(data, orient="index")
    
    def get_majority_label(annotators):
        labels = [ann["label"] for ann in annotators]
        return Counter(labels).most_common(1)[0][0]

    df["post_label"] = df["annotators"].apply(get_majority_label)
    print(df["post_label"].value_counts())

    # Clean and tokenize
    df["post_tokens"] = df["post_tokens"].apply(lambda tokens: [
        t.lower() for t in tokens if t.lower() not in stop_words and not all(c in string.punctuation for c in t)
    ])

    # Save cleaned version
    df.to_csv("cleaned_dataset.csv", index=False)
    return df

# === STEP 2: Vectorize Tokens with Word2Vec ===
def build_word_vectors(df):
    w2v = Word2Vec(sentences=df["post_tokens"], vector_size=100, window=5, min_count=2, workers=4, sg=1)
    df["post_vectors"] = df["post_tokens"].apply(lambda tokens: [w2v.wv[t] for t in tokens if t in w2v.wv])
    return df

# === STEP 3: Prepare Tensors ===
def prepare_split(df_split):
    df_filtered = df_split[df_split["post_vectors"].apply(lambda v: len(v) > 0)]
    y = torch.tensor(df_filtered["post_label"].map(label2id).values, dtype=torch.long)
    X = [torch.tensor(vectors, dtype=torch.float) for vectors in df_filtered["post_vectors"]]
    X_padded = pad_sequence(X, batch_first=True)
    return X_padded, y

# === STEP 4: LSTM Model ===
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.bidirectional = bidirectional

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(self.dropout(hidden))

# === STEP 5: Training Loop ===
def train_model(model, train_loader, val_loader, epochs, lr):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (model(X).argmax(1) == y).sum().item()
            total += y.size(0)
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

# === STEP 6: Evaluation ===
def evaluate_model(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds.extend(model(X).argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return classification_report(labels, preds, target_names=["normal", "hatespeech", "offensive"], output_dict=True)

# === STEP 7: Hyperparameter Tuning ===
def run_tuning():
    results, best_score, best_model = [], 0, None
    for hidden in [64, 128]:
        for dropout in [0.3]:
            for lr in [1e-3]:
                for batch_size in [32, 64]:
                    print(f"\n🔧 Training with H={hidden}, D={dropout}, LR={lr}, B={batch_size}")
                    model = LSTMClassifier(100, hidden, 3, dropout=dropout).to(device)
                    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
                    train_model(model, train_loader, val_loader, epochs=5, lr=lr)
                    report = evaluate_model(model, val_loader)
                    macro_f1 = report["macro avg"]["f1-score"]
                    results.append({"hidden": hidden, "dropout": dropout, "lr": lr, "batch_size": batch_size, "macro_f1": macro_f1})
                    if macro_f1 > best_score:
                        best_score = macro_f1
                        best_model = model
    pd.DataFrame(results).to_csv("tuned_model_results.csv", index=False)
    print("📊 Saved tuned results.")
    return best_model

# === MAIN ===
df = load_and_preprocess()
df = build_word_vectors(df)

# Split
label2id = {"normal": 0, "hatespeech": 1, "offensive": 2}
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["post_label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["post_label"], random_state=42)
X_train, y_train = prepare_split(train_df)
X_val, y_val = prepare_split(val_df)
X_test, y_test = prepare_split(test_df)

# Training or Tuning
if DO_TRAIN:
    print("\n📌 Training Baseline Model...")
    model = LSTMClassifier(input_dim=100, hidden_dim=128, output_dim=3, dropout=0.5).to(device)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)
    train_model(model, train_loader, val_loader, epochs=10, lr=1e-3)
    report = evaluate_model(model, val_loader)
    print("\n📊 Validation Classification Report:")
    print(pd.DataFrame(report).transpose())

if DO_TUNING:
    print("\n🚀 Running Hyperparameter Tuning...")
    best_model = run_tuning()

# --- Evaluation of both models on test set---

def evaluate_model(model, X_test, y_test, name="Model"):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for i in range(0, len(X_test), 64):  # batching for performance
            batch_X = X_test[i:i+64].to(device)
            batch_y = y_test[i:i+64].to(device)
            outputs = model(batch_X)
            predicted = torch.argmax(outputs, dim=1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(batch_y.cpu().numpy())

    # Report and metrics
    report = classification_report(labels, preds, output_dict=True, target_names=["normal", "hatespeech", "offensive"])
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    print(f"\n📊 Evaluation Report for {name}")
    print(classification_report(labels, preds, target_names=["normal", "hatespeech", "offensive"]))
    print(f"Accuracy: {acc:.4f}")

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["normal", "hatespeech", "offensive"],
                yticklabels=["normal", "hatespeech", "offensive"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(report).transpose()

# Run evaluations
baseline_eval = evaluate_model(model, X_test, y_test, name="Baseline Model")
improved_eval = evaluate_model(best_model, X_test, y_test, name="Improved Model")

# Merge for side-by-side comparison
comparison = pd.concat([baseline_eval, improved_eval], axis=1)
comparison.columns = [f"{col}_baseline" for col in baseline_eval.columns] + \
                     [f"{col}_improved" for col in improved_eval.columns]

# Display comparison table
print("\n🔍 Side-by-Side Comparison:")
print(comparison)