from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2') # Or 'BAAI/bge-small-en-v1.5'
model.save('./models/all-MiniLM-L6-v2') # Save it to a local directory
# Later load from: SentenceTransformer('./models/all-MiniLM-L6-v2')