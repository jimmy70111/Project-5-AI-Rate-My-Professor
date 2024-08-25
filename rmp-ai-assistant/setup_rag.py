from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
import json

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#pc.delete_index("rag")

# Create a Pinecone index
# can uncomment if index deleted
"""pc.create_index(
    name="rag",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)"""

# Load the review data
data = json.load(open("reviews.json"))

# Initialize the sentence-transformers model
model = SentenceTransformer('all-MiniLM-L6-v2')

processed_data = []

# Create embeddings for each review
for review in data["reviews"]:
    embedding = model.encode(review['review'])
    processed_data.append(
        {
            "values": embedding.tolist(),
            "id": review["professor"],
            "metadata": {
                "review": review["review"],
                "subject": review["subject"],
                "stars": review["stars"],
            }
        }
    )

# Insert the embeddings into the Pinecone index
index = pc.Index("rag")
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)
print(f"Upserted count: {upsert_response['upserted_count']}")

# Print index statistics
print(index.describe_index_stats())