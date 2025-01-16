import pandas as pd
from sentence_transformers import SentenceTransformer

# Load your CIF data CSV (assuming it's already preprocessed)
df = pd.read_csv('cif_data.csv')

# Example of concatenating relevant columns to form a text representation
df['embedding_input'] = (
    df['Molecule_Name'] + ' ' + 
    df['Cell_Length_A'].astype(str) + ' ' +
    df['Cell_Length_B'].astype(str) + ' ' +
    df['Cell_Length_C'].astype(str) + ' ' +
    df['Angle_Alpha'].astype(str) + ' ' +
    df['Angle_Beta'].astype(str) + ' ' +
    df['Angle_Gamma'].astype(str)
)

# Use Sentence Transformers to embed the text
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['embedding_input'].tolist())

# Save the embeddings for later use
import numpy as np
np.save('cif_embeddings.npy', embeddings)

# Optionally, save the IDs or other relevant data to match later with results
df['embedding_id'] = df.index
df.to_csv('cif_data_with_ids.csv', index=False)
