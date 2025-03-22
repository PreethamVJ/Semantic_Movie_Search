import pymongo
from dotenv import load_dotenv
import os
import requests
from huggingface_hub import login
load_dotenv()
mongo_uri = os.getenv('MONGO_URI')
client = pymongo.MongoClient(mongo_uri)
db = client.sample_mflix
collection = db.movies

hf_token=os.getenv("HUGGINGFACE_API_KEY")
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embedding(text: str) -> list[float]:

  response = requests.post(
    embedding_url,
    headers={"Authorization": f"Bearer {hf_token}"},
    json={"inputs": text})

  if response.status_code != 200:
    raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

  return response.json()

# print(generate_embedding("Hello, world!"))

# for doc in collection.find({'plot':{"$exists": True}}).limit(100):
#   doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
#   collection.replace_one({'_id': doc['_id']}, doc)


query = "imaginary characters from outer space at war"

results = collection.aggregate([
  {"$vectorSearch": {
    "queryVector": generate_embedding(query),
    "path": "plot_embedding_hf",
    "numCandidates": 100,
    "limit": 4,
    "index": "PlotSemanticSearch1",
      }}
]);

for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')

