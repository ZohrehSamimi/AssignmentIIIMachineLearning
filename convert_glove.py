from gensim.scripts.glove2word2vec import glove2word2vec

# Convert GloVe to Word2Vec format
glove2word2vec("Data/glove.840B.300d.txt", "Data/glove.840B.300d_w2v.txt")

print(" Conversion completed successfully!")
