import streamlit as st
from openai import AzureOpenAI  # Use Azure OpenAI client to match your .env
import numpy as np
import os
from dotenv import load_dotenv

# --- 1. CONFIGURATION & API KEY ---
st.set_page_config(page_title="Our Vacation Memories", page_icon="✈️", layout="centered")

# Load environment variables from your .env file
load_dotenv()

# Retrieve Azure specific variables from your .env file
azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
azure_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Extract deployment names from your .env file
chat_deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")
embedding_deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

if not azure_key or not azure_endpoint:
    st.error("Please configure your Azure OpenAI credentials in your .env file.")
    st.stop()

# Initialize the Azure OpenAI Client
client = AzureOpenAI(
    api_key=azure_key,
    api_version=azure_version,
    azure_endpoint=azure_endpoint
)

# --- 2. HELPER FUNCTIONS ---
def get_embedding(text):
    """Generates an embedding vector using Azure OpenAI."""
    response = client.embeddings.create(
        input=[text], 
        model=embedding_deployment  # Uses text-embedding-ada-002 from your .env
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    """Calculates the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_and_embed_trip_data(file_path):
    """Reads the trip data and pre-computes embeddings for entire paragraphs."""
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Split the content by double newlines into full paragraphs (individual days)
    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
        
    embedded_data = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embedded_data.append({"text": chunk, "embedding": embedding})
    return embedded_data

# --- 3. LOAD DATA ---
DATA_FILE = "Lonavla trip.txt"
if not os.path.exists(DATA_FILE):
    st.warning(f"Please create a '{DATA_FILE}' file in the directory to power the search.")
    st.stop()

trip_memories = load_and_embed_trip_data(DATA_FILE)

# --- 4. STREAMLIT UI & SEARCH ---
st.title("✈️ Our Vacation Memory Bot")
st.write("Ask anything about our trip! Enter details, locations, funny moments, or expenses.")

query = st.text_input("What do you want to find out?", placeholder="e.g., What happened on Day 2?")

if query:
    with st.spinner("Searching our memories..."):
        # 1. Embed the user's query
        query_vector = get_embedding(query)
        
        # 2. Find the most relevant paragraphs based on similarity
        scored_memories = []
        for memory in trip_memories:
            similarity = cosine_similarity(query_vector, memory["embedding"])
            scored_memories.append((similarity, memory["text"]))
            
        # Sort memories by highest similarity score
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # Baseline threshold configuration
        SIMILARITY_THRESHOLD = 0.60 
        
        # FIX: Collect top 3 matching paragraphs so Day 1 info is passed safely to the LLM
        relevant_context = [text for score, text in scored_memories[:3] if score > SIMILARITY_THRESHOLD]
        
        if not relevant_context:
            st.info("Sorry, this information is not available.")
        else:
            context_str = "\n".join(relevant_context)
            
            # 3. Formulate prompt for Azure gpt-4o-mini

# 3. Formulate prompt for Azure gpt-4o-mini
            system_prompt = (
                "You are a close, friendly assistant keeping track of memories for a group of friends. "
                "Answer the user's question completely using ONLY the provided trip context below. "
                "CRITICAL TONE INSTRUCTION: Always write your response from a first-person collective perspective "
                "using pronouns like 'We', 'Our', and 'Us' to match the perspective of the text context. "
                "Keep your answer crisp, highly detailed, accurate, and summarized in 2-4 sentences so no key details are lost. "
                "If the context doesn't truly answer the question, say 'Sorry, this information is not available.'"
            )            
            user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
            
            try:
                response = client.chat.completions.create(
                    model=chat_deployment,  # Uses gpt-4o-mini from your .env
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Fallback check
                if "sorry" in answer.lower() or "not available" in answer.lower():
                    st.info("Sorry, this information is not available.")
                else:
                    st.success("✨ Here is what I found:")
                    st.write(answer)
                    
            except Exception as e:
                st.error("An error occurred while processing your request.")

# --- 5. PHOTO GALLERY ---
st.write("---") 
st.subheader("📸 Highlights from our Trip")

# Rendering images into a 3-column grid structure
img_col1, img_col2, img_col3 = st.columns(3)

with img_col1:
    st.image("Images & Videos/Bedse caves 1.jpg", caption="Staircases to Bedse caves", use_container_width=True)
    st.image("Images & Videos/Grocery shopping.jpg", caption="Fruits shopping on our return trip at Vashi market", use_container_width=True)
    st.image("Images & Videos/Vashi lunch.jpg", caption="Lunch at Vashi en route to Lonavla", use_container_width=True)

with img_col2:
    st.image("Images & Videos/Bedse Caves group.jpg", caption="Our group photo at Bedse caves", use_container_width=True)
    st.image("Images & Videos/Mayuresh Lunch.jpg", caption="Lunch at Mayuresh at Talegaon", use_container_width=True)

with img_col3:
    st.image("Images & Videos/Chaitanya meditation spot.jpg", caption="Chaitanya found a place to meditate", use_container_width=True)
    st.image("Images & Videos/Rajmachi.jpg", caption="A small halt at Rajmachi on our return trip", use_container_width=True)
    