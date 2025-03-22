from gensim.models import KeyedVectors

# Load only the first 500,000 words (adjust as needed)
word2vec = KeyedVectors.load_word2vec_format("Data/glove.840B.300d_w2v.txt", binary=False, limit=500000)

# Save the reduced model
word2vec.save("Data/word2vec.model")

print(" Reduced Word2Vec model saved successfully!")
