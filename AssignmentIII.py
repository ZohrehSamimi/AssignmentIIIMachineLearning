import pandas as pd
import json
from collections import Counter

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

# Display updated DataFrame
print(df[["post_id", "post_label", "post_tokens"]].head())

#This way, you won’t need to process dataset.json again!
df.to_csv("cleaned_dataset.csv", index=False)
print("Dataset saved as cleaned_dataset.csv")

