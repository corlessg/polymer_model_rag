import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Load your CIF data CSV (assuming it's already preprocessed)
df = pd.read_csv('./cif_data_streamed.csv')

# Example of concatenating relevant columns to form a text representation
df['embedding_input'] = (
    df['Molecule_Name'] + ' ' + 
    df['Label'].astype(str) + ' ' +
    df['Cell_Length_A'].astype(str) + ' ' +
    df['Cell_Length_B'].astype(str) + ' ' +
    df['Cell_Length_C'].astype(str) + ' ' +
    df['Cell_Angle_Alpha'].astype(str) + ' ' +
    df['Cell_Angle_Beta'].astype(str) + ' ' +
    df['Cell_Angle_Gamma'].astype(str) + ' ' +
    df['Fractional_X'].astype(str) + ' ' +
    df['Fractional_Y'].astype(str) + ' ' +
    df['Fractional_Z'].astype(str) 

)

# Use Sentence Transformers to embed the text
model = SentenceTransformer('allenai/scibert_scivocab_uncased')
model = model.to('cuda')
embeddings = model.encode(df['embedding_input'].tolist())

# Save the embeddings for later use
np.save('./cif_embeddings.npy', embeddings)

# Optionally, save the IDs or other relevant data to match later with results
df['embedding_id'] = df.index
df.to_csv('./cif_data_with_ids.csv', index=False)


# Define FAISS index (you can experiment with different index types for efficiency)
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
index.add(embeddings)

# Optionally, store the FAISS index to disk for later use
faiss.write_index(index, "./cif_data_index.faiss")

