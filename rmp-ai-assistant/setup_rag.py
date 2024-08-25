from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
import requests
import os
import json

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create a Pinecone index
pc.create_index(
    name="rag",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

# Load the review data
data = json.load(open("reviews.json"))

processed_data = []

# Define headers
headers = {
    "Authorization": f"Bearer {os.getenv('LLAMA_API_KEY')}",  # Replace with your actual API key
    "Content-Type": "application/json"
}

# Create embeddings for each review
for review in data["reviews"]:
    payload = {
        "input": review['review'],
        "model": "meta-llama/llama-3.1-8b-instruct:free"
    }

    response = requests.post("https://openrouter.ai/api/v1/embeddings", headers=headers, json=payload)

    # Check for successful response
    if response.status_code == 200:
        response_data = response.json()
        embedding = response_data['data'][0]['embedding']  # Ensure this is the correct path in the response JSON
        
        processed_data.append(
            {
                "values": embedding,
                "id": review["professor"],
                "metadata": {
                    "review": review["review"],
                    "subject": review["subject"],
                    "stars": review["stars"],
                }
            }
        )
    else:
        print(f"Error creating embedding for review: {review['review']}")
        print(f"Response Code: {response.status_code}, Response Text: {response.text}")

# Insert the embeddings into the Pinecone index
index = pc.Index("rag")
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)
print(f"Upserted count: {upsert_response['upserted_count']}")

# Print index statistics
print(index.describe_index_stats())
