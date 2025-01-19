import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from transformers import pipeline



index = faiss.read_index("./cif_data_index.faiss")
model = SentenceTransformer('allenai/scibert_scivocab_uncased')
model = model.to('cuda')

df = pd.read_csv('./cif_data_with_ids.csv')
# Encode user query
query = "What are the properties of polystyrene?"
query_embedding = model.encode([query])

# Retrieve the most similar CIF data entries using FAISS
k = 5  # Number of similar results to retrieve
D, I = index.search(query_embedding, k)

# Fetch the corresponding CIF data rows based on the indices
retrieved_data = df.iloc[I[0]]

print('retreived this from embedded query')
print(retrieved_data)

# Construct the context to pass into the language model
context = f"""
You are a scientific assistant specialized in crystallography. Below is data retrieved from a CIF file:

{retrieved_data[['Molecule_Name', 'Cell_Length_A', 'Cell_Length_B', 'Cell_Length_C', 'Cell_Angle_Alpha', 'Cell_Angle_Beta', 'Cell_Angle_Gamma']].to_string(index=False)}

User Question: {query}
"""
print('context that was retrieved:')
print(context)

# Initialize a text generation pipeline (using any model like T5 or GPT-2)
generator = pipeline('text-generation', model='allenai/scibert_scivocab_uncased', device=0)

# Generate the answer based on the context
generated_answer = generator(context, max_new_tokens=200)
print('answer that was generated with prompt and context:')
print(generated_answer[0]['generated_text'])
