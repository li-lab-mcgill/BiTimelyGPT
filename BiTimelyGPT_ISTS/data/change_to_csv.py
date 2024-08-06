import pandas as pd
import gensim

# Load the model
model = gensim.models.Word2Vec.load("./phecode_word2vec.model")

# Extract PheCodes and their embeddings
phecodes = list(model.wv.index_to_key)
embeddings = [model.wv[phecode] for phecode in phecodes]

# Create a DataFrame
embedding_df = pd.DataFrame(embeddings, index=phecodes)

# Convert index to integer
embedding_df.index = embedding_df.index.astype(int)

# Save to CSV
embedding_df.to_csv("./phecode_embeddings.csv")
