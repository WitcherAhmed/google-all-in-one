import os
import json
from google import genai  # Changed from google.generativeai
from google.genai import types

# Setup pathing correctly
working_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(working_dir, "config.json")

with open(config_file_path, "r") as f:
    config_data = json.load(f)

GOOGLE_API_KEY = config_data['GOOGLE_API_KEY']

# Initialize the new Client
client = genai.Client(api_key=GOOGLE_API_KEY)

# 1. Text-to-Text Response
def gemini_pro_response(user_prompt):
    # 'models/' prefix is no longer strictly required in the name string
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=user_prompt
    )
    return response.text

# 2. Vision/Multimodal Response
def gemini_pro_vision_response(prompt, image):
    # In the new SDK, you just pass the image in the list
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt, image]
    )
    return response.text

# 3. Embeddings Response
def embeddings_model_response(input_text):
    # Use the new embed_content method on the client
    response = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=input_text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    # The new SDK returns a list of embeddings; we take the first one
    return response.embeddings[0].values

# Simple debug loop to see available models
for m in client.models.list():
    print(f"Model: {m.name}")
