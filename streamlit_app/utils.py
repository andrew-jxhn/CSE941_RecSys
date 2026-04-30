import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Base paths
BASE_PATH = Path(__file__).parent / "outputs"

def load_phase1_data():
    """Load Phase 1 data and metadata"""
    with open(BASE_PATH / "phase1" / "processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    return data['reviews'], data['metadata']

def load_phase2_embeddings():
    """Load CLIP embeddings"""
    with open(BASE_PATH / "phase2" / "clip_embeddings_dict.pkl", "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

def load_phase3_mappings():
    """Load ID mappings"""
    with open(BASE_PATH / "phase3" / "id_mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    return mappings

def load_phase4_embeddings():
    """Load GNN embeddings"""
    user_emb = np.load(BASE_PATH / "phase4" / "user_embeddings.npy")
    item_emb = np.load(BASE_PATH / "phase4" / "item_embeddings.npy")
    return user_emb, item_emb

def load_phase5_coldstart():
    """Load cold-start embeddings"""
    with open(BASE_PATH / "phase5" / "coldstart_embeddings.pkl", "rb") as f:
        coldstart = pickle.load(f)
    return coldstart

def load_phase6_explanations():
    """Load LLM explanations"""
    explanations = pd.read_csv(BASE_PATH / "phase6" / "sample_explanations.csv")
    return explanations

def get_item_info(item_id, meta_df):
    """Get item details"""
    item_data = meta_df[meta_df['parent_asin'] == item_id]
    
    if len(item_data) == 0:
        return {
            'item_id': item_id,
            'title': 'Beauty Product',
            'category': 'Beauty',
            'avg_rating': None,
            'price': None,
            'images': []
        }
    
    item = item_data.iloc[0]
    return {
        'item_id': item_id,
        'title': item.get('title', 'Beauty Product'),
        'category': item.get('main_category', 'Beauty'),
        'avg_rating': item.get('average_rating', None),
        'price': item.get('price', None),
        'images': item.get('images', [])
    }

def get_user_history(user_id, reviews_df, meta_df, top_n=5):
    """Get user purchase history"""
    user_reviews = reviews_df[reviews_df['user_id'] == user_id].sort_values('rating', ascending=False).head(top_n)
    
    history = []
    for _, review in user_reviews.iterrows():
        item_id = review['parent_asin']
        item_info = get_item_info(item_id, meta_df)
        
        history.append({
            'item_id': item_id,
            'title': item_info['title'],
            'rating': review['rating'],
            'images': item_info['images']
        })
    
    return history

def find_similar_items_visual(target_item, clip_embeddings, top_k=10):
    """Find visually similar items using CLIP"""
    if target_item not in clip_embeddings:
        return []
    
    target_emb = clip_embeddings[target_item].reshape(1, -1)
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = []
    for item, emb in clip_embeddings.items():
        if item != target_item:
            sim = cosine_similarity(target_emb, emb.reshape(1, -1))[0, 0]
            similarities.append((item, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def find_similar_items_collaborative(target_item, item_embeddings, item_to_idx, idx_to_item, top_k=10):
    """Find similar items using GNN embeddings"""
    if target_item not in item_to_idx:
        return []
    
    target_idx = item_to_idx[target_item]
    target_emb = item_embeddings[target_idx]
    
    # Compute similarities
    similarities = []
    for idx, emb in enumerate(item_embeddings):
        if idx != target_idx:
            sim = np.dot(target_emb, emb) / (np.linalg.norm(target_emb) * np.linalg.norm(emb))
            similarities.append((idx_to_item[idx], sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def get_recommendations_for_user(user_id, user_embeddings, item_embeddings, user_to_idx, idx_to_item, top_k=10):
    """Get recommendations for a user"""
    if user_id not in user_to_idx:
        return []
    
    user_idx = user_to_idx[user_id]
    user_emb = user_embeddings[user_idx]
    
    # Compute scores
    scores = np.dot(item_embeddings, user_emb)
    
    # Get top-k
    top_indices = np.argsort(scores)[::-1][:top_k]
    recommendations = [(idx_to_item[idx], scores[idx]) for idx in top_indices]
    
    return recommendations

def search_products(query, meta_df):
    """Search products by name"""
    if 'title' not in meta_df.columns:
        return []
    
    query_lower = query.lower()
    matches = meta_df[meta_df['title'].str.lower().str.contains(query_lower, na=False)]
    
    results = []
    for _, item in matches.head(20).iterrows():
        results.append({
            'item_id': item['parent_asin'],
            'title': item.get('title', 'Beauty Product'),
            'category': item.get('main_category', 'Beauty'),
            'avg_rating': item.get('average_rating', None),
            'images': item.get('images', [])
        })
    
    return results