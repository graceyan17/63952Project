import google.generativeai as genai
import json
import time
import numpy as np
import pandas as pd

import os

print("Current directory:", os.getcwd())

df = pd.read_csv("datasets/joined_complete_info.csv")
df = df.drop_duplicates(subset=['join_title', 'book_desc'], keep='first')
df.set_index('join_title', inplace=True)
print('columns', df.columns)

# Configure Gemini
# genai.configure(api_key='INSERT API KEY HERE')
# print("Available models:")
# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)
# model = genai.GenerativeModel('models/gemini-2.0-flash')

def get_gemini_recommendations(user_high_rated_books, candidate_books, top_k=50):
    """
    Get recommendations from Gemini
    """
    print(user_high_rated_books)
    if len(candidate_books) > 2000:
        import random
        candidate_books = random.sample(candidate_books, 2000)
    
    prompt = f"""You are an expert book recommendation system.

A user has enjoyed and highly rated these books:
{chr(10).join(f"- {book}" for book in user_high_rated_books)}

From the following candidate books, recommend the top {top_k} books that this user would most likely enjoy. Consider genre, themes, writing style, and author similarities.

Candidate books:
{chr(10).join(f"- {book}" for book in candidate_books)}

CRITICAL INSTRUCTIONS:
1. Return ONLY the EXACT FULL BOOK TITLES from the candidate list above
2. Do NOT return numbers, indices, or shortened titles
3. Return a JSON array with exactly {top_k} complete book titles
4. Rank them by relevance (most relevant first)

Example correct format: ["the great gatsby", "to kill a mockingbird", "1984"]
Example WRONG format: ["1", "2", "3"] or ["gatsby", "mockingbird"]

Your response (JSON array only, no explanations):"""
    
    try:
        response = model.generate_content(prompt)
        # Clean up response text
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        
        recommendations = json.loads(response_text.strip())
        return recommendations[:top_k]
    
    except Exception as e:
        print(f"Error: {e}")
        print(f"Response text: {response.text if 'response' in locals() else 'No response'}")
        return []
    
import subprocess
import json

def get_llama_recommendations(user_high_rated_books, candidate_books, top_k=50):
    """
    Generate book recommendations using Llama (Ollama backend)
    """
    if len(candidate_books) > 2000:
        import random
        candidate_books = random.sample(candidate_books, 2000)

    prompt = f"""You are an expert literary recommendation engine.

A user has highly rated these books:
{chr(10).join(f"- {book}" for book in user_high_rated_books)}

From the list below, recommend the top {top_k} books this user would MOST LIKELY enjoy.

Candidate books:
{chr(10).join(f"- {book}" for book in candidate_books)}

CRITICAL INSTRUCTIONS:
1. Return ONLY the EXACT FULL BOOK TITLES found in the candidate list.
2. NO commentary, NO numbering, NO shortened titles.
3. Return ONLY a JSON array with exactly {top_k} titles.
4. Order them most relevant first.

Example correct format: ["The Great Gatsby", "1984", "To Kill a Mockingbird"]

JSON only:
"""

    try:
        # call Ollama CLI
        result = subprocess.run(
            ["ollama", "run", "llama3.1", prompt],
            capture_output=True,
            text=True
        )

        response_text = result.stdout.strip()

        # Try to isolate JSON
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
                
        # load JSON
        recommendations = json.loads(response_text)

        # ensure correct length
        return recommendations[:top_k]

    except Exception as e:
        print("LLAMA ERROR:", e)
        print("RAW RESPONSE:", response_text)
        return []

# Generate recommendations using Gemini
recommendations = {}
k = 50

unmasked_df = pd.read_csv("datasets/train_edges.csv")

user_books = unmasked_df.groupby('User-ID').apply(
    lambda x: list(zip(x['join_title'], x['Book-Rating']))
).to_dict()

for i, (user_id, books_ratings) in enumerate(user_books.items()):
    print(f"Processing user {user_id} ({i+1}/{len(user_books)})")
    
    # Get high-rated books for this user
    high_rated_unmasked = [join_title for join_title, rating in books_ratings if rating > 7]
    
    if len(high_rated_unmasked) == 0:
        print(f"  User {user_id} has no highly rated books, skipping")
        continue
    
    print(f"  User {user_id} has rated {len(high_rated_unmasked)} books highly")
    
    # Get unmasked titles to exclude from recommendations
    unmasked_titles = {title for title, _ in books_ratings}
    
    # Get all candidate books (exclude unmasked)
    candidate_books = [title for title in df.index if title not in unmasked_titles]
    
    print(f"  Candidate pool: {len(candidate_books)} books")
    
    # Get recommendations from Llama
    user_recommendations = get_llama_recommendations(
        high_rated_unmasked, 
        candidate_books, 
        top_k=k
    )
    
    if len(user_recommendations) > 0:
        recommendations[user_id] = user_recommendations
        print(f"  Got {len(user_recommendations)} recommendations")
    else:
        print(f"  Failed to get recommendations for user {user_id}")
    
    # Rate limiting - Gemini free tier allows 60 requests/minute
    time.sleep(1)
    
    if (i + 1) % 10 == 0:
        with open('datasets/llama_recommendations_checkpoint.json', 'w') as f:
            json.dump({str(k): v for k, v in recommendations.items()}, f, indent=2)
        print(f"  Checkpoint saved after {i+1} users")

# Save final recommendations
with open('datasets/llama_recommendations.json', 'w') as f:
    json.dump({str(k): v for k, v in recommendations.items()}, f, indent=2)

print(f"\nCompleted! Generated recommendations for {len(recommendations)} users")