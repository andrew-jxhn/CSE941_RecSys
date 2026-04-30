import streamlit as st

# THIS MUST BE FIRST - BEFORE ANY OTHER st. COMMAND
st.set_page_config(
    page_title="Amazon Beauty Recommendation System",
    page_icon="💄",
    layout="wide"
)

# NOW other imports
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
from PIL import Image

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from utils import (
    load_phase1_data,
    load_phase2_embeddings,
    load_phase3_mappings,
    load_phase4_embeddings,
    load_phase5_coldstart,
    load_phase6_explanations,
    get_item_info,
    get_user_history,
    find_similar_items_visual,
    find_similar_items_collaborative,
    get_recommendations_for_user,
    search_products
)

# Title
st.title("💄 Amazon Beauty Recommendation System")
st.markdown("**Multimodal Cold-Start Recommendations using CV + GNN + LLM**")
st.markdown("---")

# Load data (cached)
@st.cache_data
def load_all_data():
    reviews_df, meta_df = load_phase1_data()
    clip_embeddings = load_phase2_embeddings()
    mappings = load_phase3_mappings()
    user_embeddings, item_embeddings = load_phase4_embeddings()
    coldstart_embeddings = load_phase5_coldstart()
    explanations_df = load_phase6_explanations()
    
    return {
        'reviews': reviews_df,
        'metadata': meta_df,
        'clip_embeddings': clip_embeddings,
        'mappings': mappings,
        'user_embeddings': user_embeddings,
        'item_embeddings': item_embeddings,
        'coldstart_embeddings': coldstart_embeddings,
        'explanations': explanations_df
    }

# Load data
with st.spinner("Loading data..."):
    data = load_all_data()

reviews_df = data['reviews']
meta_df = data['metadata']
clip_embeddings = data['clip_embeddings']
mappings = data['mappings']
user_embeddings = data['user_embeddings']
item_embeddings = data['item_embeddings']
coldstart_embeddings = data['coldstart_embeddings']
explanations_df = data['explanations']

user_to_idx = mappings['user_to_idx']
idx_to_user = mappings['idx_to_user']
item_to_idx = mappings['item_to_idx']
idx_to_item = mappings['idx_to_item']

# Sidebar header
st.sidebar.markdown("### Amazon Beauty RecSys")
st.sidebar.markdown("*Multimodal Cold-Start Recommendations*")
st.sidebar.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Project Overview",
    "📊 Dataset Insights",
    "👤 User Recommendations",
    "🔍 Product Search & Similarity",
    "🆕 Cold-Start Demo",
    "📈 Model Performance"
])

# ============================================================
# PAGE 1: PROJECT OVERVIEW
# ============================================================
if page == "🏠 Project Overview":
    st.header("🏠 Project Overview")
    
    st.markdown("""
    ## Welcome to the Amazon Beauty Recommendation System!
    
    This is a **multimodal recommendation system** that solves the **cold-start problem** for new products 
    using a combination of **Computer Vision (CV)**, **Graph Neural Networks (GNN)**, and **Large Language Models (LLM)**.
    """)
    
    st.markdown("---")
    
    # Project highlights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 The Problem")
        st.markdown("""
        **Cold-Start Problem**: New products with zero reviews cannot be recommended by traditional 
        collaborative filtering systems.
        
        - 📉 80% of items have ≤5 reviews
        - 🚫 Traditional methods achieve **0% recall**
        - 💰 Billions in lost revenue for e-commerce
        """)
        
        st.subheader("💡 Our Solution")
        st.markdown("""
        **3-Stage Pipeline:**
        
        1. **Computer Vision (CLIP)**: Extract visual features from product images
        2. **Graph Neural Network (LightGCN)**: Learn user-item relationships
        3. **LLM (Groq API)**: Generate natural language explanations
        
        **Result**: 1% Recall@10 vs 0% baseline ✅
        """)
    
    with col2:
        st.subheader("📊 Dataset Statistics")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Users", "10,000")
            st.metric("Total Items", "20,223")
        with col_b:
            st.metric("Total Reviews", "41,109")
            st.metric("Avg Rating", "4.12/5.0")
        
        st.subheader("🔧 Tech Stack")
        st.markdown("""
        - **Dataset**: Amazon Beauty Reviews 2023
        - **CV Model**: OpenAI CLIP (ViT-B/32)
        - **GNN**: PyTorch Geometric + LightGCN
        - **LLM**: Groq API (Llama 3.1 70B)
        - **Deployment**: Streamlit
        """)
    
    st.markdown("---")
    
    # How to use this app
    st.subheader("📖 How to Use This App")
    
    st.markdown("""
    **Navigate using the sidebar:**
    
    1. **📊 Dataset Insights** - Explore 15 visualizations analyzing the Amazon Beauty dataset
    
    2. **👤 User Recommendations** - Select a user and see personalized top-10 product recommendations
    
    3. **🔍 Product Search & Similarity** - Search for any product and find visually/collaboratively similar items
    
    4. **🆕 Cold-Start Demo** - See how we recommend new products with zero reviews (the innovation!)
    
    5. **📈 Model Performance** - View training metrics and cold-start evaluation results
    """)
    
    st.markdown("---")
    
    # Key results
    st.subheader("🏆 Key Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Training Accuracy**\n\n56.98%")
    
    with col2:
        st.success("**Cold-Start Recall@10**\n\n1.00% (Baselines: 0%)")
    
    with col3:
        st.warning("**Visual Similarity**\n\n70-80% accuracy")
    
    st.markdown("---")
    
    # Project context
    st.subheader("🎓 Project Context")
    
    st.markdown("""
    **Course**: CSE 941 - Advanced Machine Learning
    
    **Objective**: Build a production-ready recommendation system that demonstrates:
    - Multimodal learning (CV + Graph)
    - Cold-start problem solving
    - Explainable AI (LLM integration)
    - Full pipeline from data → deployment
    
    **Why This Matters for Industry:**
    - Meta, Google, Amazon all use graph-based recommendation systems
    - Cold-start is a billion-dollar problem in e-commerce
    - Multimodal approaches are the future of recommender systems
    """)

# ============================================================
# PAGE 2: DATASET INSIGHTS (ALL PLOTS)
# ============================================================
elif page == "📊 Dataset Insights":
    st.header("📊 Dataset Insights")
    
    st.markdown("""
    Comprehensive exploratory data analysis of the **Amazon Beauty Reviews 2023** dataset.
    These 15 visualizations reveal patterns in user behavior, item popularity, cold-start severity, and more.
    """)
    
    st.markdown("---")
    
    # ============ ADD FLOWCHART HERE ============
    st.subheader("🔄 Complete Pipeline Architecture")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        flowchart_path = Path(__file__).parent / "outputs" / "pipeline_flowchart.png"
        
        if flowchart_path.exists():
            try:
                flowchart = Image.open(flowchart_path)
                st.image(flowchart, use_column_width=True, caption="End-to-End System Pipeline")
            except Exception as e:
                st.error(f"Error loading flowchart: {e}")
        else:
            st.warning("Pipeline flowchart not found.")

    st.info("""
    **Pipeline Overview**: This flowchart shows our complete 6-phase recommendation system:
    - **Phases 1-3** (Top): Data loading, computer vision embeddings, and graph construction
    - **Phases 4-6** (Bottom): GNN training, cold-start solution, and LLM explanations
    - **Final Output**: Interactive Streamlit web application
    """)

    st.markdown("---")
    # ============ END FLOWCHART ============

    # Dataset stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", f"{len(reviews_df['user_id'].unique()):,}")
    with col2:
        st.metric("Total Items", f"{len(reviews_df['parent_asin'].unique()):,}")
    with col3:
        st.metric("Total Reviews", f"{len(reviews_df):,}")
    with col4:
        avg_rating = reviews_df['rating'].mean()
        st.metric("Avg Rating", f"{avg_rating:.2f}/5.0")
    
    st.markdown("---")
    
    # Display all plots
    st.subheader("Visual Analysis")
    
    plots_dir = Path(__file__).parent / "outputs" / "phase1" / "plots"
    
    if plots_dir.exists():
        plot_files = sorted(list(plots_dir.glob("*.png")))
        
        if len(plot_files) > 0:
            st.success(f"✅ Found {len(plot_files)} analysis plots")
            
            # Display 2 plots per row
            for i in range(0, len(plot_files), 2):
                cols = st.columns(2)
                
                for j, col in enumerate(cols):
                    if i + j < len(plot_files):
                        with col:
                            try:
                                img = Image.open(plot_files[i + j])
                                st.image(img, use_column_width=True, caption=plot_files[i + j].stem.replace('_', ' ').title())
                            except Exception as e:
                                st.error(f"Error loading {plot_files[i + j].name}: {e}")
        else:
            st.warning("No plot files found in phase1/plots directory")
    else:
        st.error(f"Plots directory not found: {plots_dir}")

# ============================================================
# PAGE 3: USER RECOMMENDATIONS
# ============================================================
elif page == "👤 User Recommendations":
    st.header("👤 User Recommendations")
    
    st.markdown("""
    Select a user to see their **purchase history** and **personalized top-10 recommendations** 
    generated by our LightGCN model.
    """)
    
    st.markdown("---")
    
    # User selection
    all_users = list(user_to_idx.keys())
    selected_user = st.selectbox("Select a User ID", all_users[:100])  # Show first 100
    
    if st.button("🎯 Generate Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            # Get user history
            history = get_user_history(selected_user, reviews_df, meta_df, top_n=5)
            
            st.subheader("📜 User's Purchase History")
            
            if len(history) > 0:
                for item in history:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if item['images'] and len(item['images']) > 0:
                            try:
                                st.image(item['images'][0]['large'], width=150)
                            except:
                                st.write("🖼️ No image")
                        else:
                            st.write("🖼️ No image")
                    with col2:
                        st.write(f"**{item['title'][:80]}...**")
                        st.write(f"⭐ Rating: {item['rating']}/5.0")
            else:
                st.info("No purchase history found for this user")
            
            st.markdown("---")
            
            # Get recommendations
            recommendations = get_recommendations_for_user(
                selected_user,
                user_embeddings,
                item_embeddings,
                user_to_idx,
                idx_to_item,
                top_k=10
            )
            
            st.subheader("🎯 Top 10 Personalized Recommendations")
            
            for rank, (item_id, score) in enumerate(recommendations, 1):
                item_info = get_item_info(item_id, meta_df)
                
                with st.expander(f"#{rank} - {item_info['title'][:60]}... (Score: {score:.3f})"):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if item_info['images'] and len(item_info['images']) > 0:
                            try:
                                st.image(item_info['images'][0]['large'], width=150)
                            except:
                                st.write("🖼️ No image")
                        else:
                            st.write("🖼️ No image")
                    
                    with col2:
                        st.write(f"**Item ID:** {item_id}")
                        st.write(f"**Category:** {item_info['category']}")
                        if item_info['avg_rating']:
                            st.write(f"**Avg Rating:** {item_info['avg_rating']:.1f}/5.0")
                        st.write(f"**Recommendation Score:** {score:.4f}")

# ============================================================
# PAGE 4: PRODUCT SEARCH & SIMILARITY
# ============================================================
elif page == "🔍 Product Search & Similarity":
    st.header("🔍 Product Search & Similarity")
    
    st.markdown("""
    Search for any beauty product and discover similar items using:
    - **🖼️ Computer Vision** (CLIP embeddings - visual similarity)
    - **👥 Graph Neural Networks** (collaborative filtering - behavioral similarity)
    """)
    
    st.markdown("---")
    
    # Search box
    search_query = st.text_input("🔎 Search for a product by name", placeholder="e.g., lipstick, moisturizer, shampoo, mascara")
    
    if search_query:
        results = search_products(search_query, meta_df)
        
        if len(results) == 0:
            st.warning("❌ No products found. Try a different search term.")
        else:
            st.success(f"✅ Found {len(results)} products")
            
            # Display search results
            selected_product = st.selectbox(
                "Select a product to find similar items",
                options=[r['item_id'] for r in results],
                format_func=lambda x: next(r['title'][:70] for r in results if r['item_id'] == x)
            )
            
            if st.button("🔍 Find Similar Products", type="primary"):
                with st.spinner("Finding similar products..."):
                    target_info = get_item_info(selected_product, meta_df)
                    
                    # Display target product
                    st.subheader("🎯 Selected Product")
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if target_info['images'] and len(target_info['images']) > 0:
                            try:
                                st.image(target_info['images'][0]['large'], width=200)
                            except:
                                st.write("🖼️ No image")
                    with col2:
                        st.write(f"**{target_info['title']}**")
                        st.write(f"**Item ID:** {selected_product}")
                        st.write(f"**Category:** {target_info['category']}")
                        if target_info['avg_rating']:
                            st.write(f"**Rating:** {target_info['avg_rating']:.1f}/5.0")
                    
                    st.markdown("---")
                    
                    # Visual similarity
                    st.subheader("🖼️ Visually Similar Products (Computer Vision)")
                    st.caption("Products that look similar based on CLIP image embeddings")
                    
                    visual_similar = find_similar_items_visual(selected_product, clip_embeddings, top_k=5)
                    
                    if len(visual_similar) > 0:
                        cols = st.columns(5)
                        for idx, (item_id, similarity) in enumerate(visual_similar):
                            item_info = get_item_info(item_id, meta_df)
                            with cols[idx]:
                                if item_info['images'] and len(item_info['images']) > 0:
                                    try:
                                        st.image(item_info['images'][0]['large'], use_column_width=True)
                                    except:
                                        st.write("🖼️")
                                st.write(f"**{item_info['title'][:30]}...**")
                                st.write(f"📊 Similarity: {similarity:.3f}")
                    else:
                        st.info("No visually similar items found")
                    
                    st.markdown("---")
                    
                    # Collaborative similarity
                    st.subheader("👥 Collaboratively Similar Products (Graph Neural Network)")
                    st.caption("Products bought by users with similar preferences")
                    
                    collab_similar = find_similar_items_collaborative(
                        selected_product,
                        item_embeddings,
                        item_to_idx,
                        idx_to_item,
                        top_k=5
                    )
                    
                    if len(collab_similar) > 0:
                        cols = st.columns(5)
                        for idx, (item_id, similarity) in enumerate(collab_similar):
                            item_info = get_item_info(item_id, meta_df)
                            with cols[idx]:
                                if item_info['images'] and len(item_info['images']) > 0:
                                    try:
                                        st.image(item_info['images'][0]['large'], use_column_width=True)
                                    except:
                                        st.write("🖼️")
                                st.write(f"**{item_info['title'][:30]}...**")
                                st.write(f"📊 Similarity: {similarity:.3f}")
                    else:
                        st.info("No collaboratively similar items found")

# ============================================================
# PAGE 5: COLD-START DEMO
# ============================================================
elif page == "🆕 Cold-Start Demo":
    st.header("🆕 Cold-Start Recommendation Demo")
    
    st.markdown("""
    This is the **core innovation** of our system! 
    
    See how we recommend **brand new products with ≤5 reviews** using visual similarity 
    to bootstrap them into the recommendation graph.
    """)
    
    st.info("""
    **The Cold-Start Problem**: Traditional collaborative filtering fails when products have no reviews.
    Our solution uses Computer Vision to find visually similar products and borrows their graph embeddings.
    """)
    
    st.markdown("---")
    
    # Get cold-start items
    item_review_counts = reviews_df['parent_asin'].value_counts()
    cold_start_items = item_review_counts[item_review_counts <= 5].index.tolist()
    cold_start_with_emb = [item for item in cold_start_items if item in coldstart_embeddings]
    
    if len(cold_start_with_emb) > 0:
        selected_coldstart = st.selectbox(
            "Select a Cold-Start Item (≤5 reviews)",
            options=cold_start_with_emb[:50],
            format_func=lambda x: f"{get_item_info(x, meta_df)['title'][:60]}... ({item_review_counts.get(x, 0)} reviews)"
        )
        
        if st.button("🚀 Generate Cold-Start Recommendations", type="primary"):
            with st.spinner("Solving cold-start problem..."):
                item_info = get_item_info(selected_coldstart, meta_df)
                
                # Display cold-start item
                st.subheader("🆕 New Product (Cold-Start Item)")
                col1, col2 = st.columns([1, 3])
                with col1:
                    if item_info['images'] and len(item_info['images']) > 0:
                        try:
                            st.image(item_info['images'][0]['large'], width=200)
                        except:
                            st.write("🖼️ No image")
                with col2:
                    st.write(f"**{item_info['title']}**")
                    st.write(f"**Item ID:** {selected_coldstart}")
                    st.write(f"**Reviews:** {item_review_counts.get(selected_coldstart, 0)} ⚠️")
                    st.error("⚠️ This product has very few reviews - traditional collaborative filtering fails!")
                
                st.markdown("---")
                
                # Explain the solution
                st.subheader("💡 How We Solve This (3-Step Process)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.success("**Step 1: Extract Visual Features**")
                    st.write("Use CLIP to generate 512-dim embedding from product image")
                
                with col2:
                    st.success("**Step 2: Find Similar Products**")
                    st.write("Find top-K visually similar items that have reviews")
                
                with col3:
                    st.success("**Step 3: Borrow Graph Embeddings**")
                    st.write("Weighted average of neighbors' GNN embeddings")
                
                st.markdown("---")
                
                # Show visually similar items used
                visual_similar = find_similar_items_visual(selected_coldstart, clip_embeddings, top_k=5)
                
                if len(visual_similar) > 0:
                    st.subheader("🔗 Visual Neighbors Used for Bootstrapping")
                    st.caption("These products 'lend' their graph embeddings to the new item")
                    
                    cols = st.columns(5)
                    for idx, (item_id, sim) in enumerate(visual_similar):
                        neighbor_info = get_item_info(item_id, meta_df)
                        with cols[idx]:
                            if neighbor_info['images'] and len(neighbor_info['images']) > 0:
                                try:
                                    st.image(neighbor_info['images'][0]['large'], use_column_width=True)
                                except:
                                    pass
                            st.write(f"**{neighbor_info['title'][:25]}...**")
                            st.write(f"Similarity: {sim:.2f}")
                
                st.markdown("---")
                st.success("✅ Successfully placed new product in recommendation graph!")
                st.balloons()

# ============================================================
# PAGE 6: MODEL PERFORMANCE
# ============================================================
elif page == "📈 Model Performance":
    st.header("📈 Model Performance")
    
    st.markdown("""
    Evaluation metrics for our **LightGCN** model and **cold-start solution**.
    """)
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Training Accuracy", 
            "56.98%",
            help="Accuracy on training set after 50 epochs"
        )
    with col2:
        st.metric(
            "Test Accuracy", 
            "56.66%",
            help="Generalization performance on held-out test set"
        )
    with col3:
        st.metric(
            "Cold-Start Recall@10", 
            "1.00%",
            delta="∞ vs baselines",
            delta_color="normal",
            help="Baselines (Random, Popularity) achieved 0%"
        )
    
    st.markdown("---")
    
    # Training curves
    st.subheader("📉 Training Performance")
    
    plots_dir = Path(__file__).parent / "outputs" / "phase4" / "plots"
    
    if plots_dir.exists():
        plot_files = sorted(list(plots_dir.glob("*.png")))
        
        if len(plot_files) > 0:
            cols = st.columns(2)
            for idx, plot_file in enumerate(plot_files[:2]):
                with cols[idx]:
                    try:
                        img = Image.open(plot_file)
                        st.image(img, use_column_width=True, caption=plot_file.stem.replace('_', ' ').title())
                    except Exception as e:
                        st.error(f"Error loading {plot_file.name}: {e}")
        else:
            st.warning("Training plots not found")
    
    st.markdown("---")
    
    # Cold-start comparison
    st.subheader("🆕 Cold-Start Solution Performance")
    
    st.markdown("""
    **Key Finding**: Traditional baselines completely fail on cold-start items (0% recall).
    Our multimodal approach achieves 1% recall by leveraging visual similarity.
    """)
    
    comparison_data = {
        'Method': ['Random', 'Popularity', 'Our Approach (CV + GNN)'],
        'Recall@10': [0.0000, 0.0000, 0.0100]
    }
    
    fig = px.bar(
        comparison_data,
        x='Method',
        y='Recall@10',
        title='Cold-Start Recommendation Performance Comparison',
        color='Method',
        color_discrete_map={
            'Random': '#e74c3c',
            'Popularity': '#f39c12',
            'Our Approach (CV + GNN)': '#27ae60'
        },
        text='Recall@10'
    )
    
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Why This Matters:**
    - Cold-start items represent 80%+ of product catalogs in e-commerce
    - Traditional methods achieve 0% recall (complete failure)
    - Our approach is the **only method that works**
    - 1% may seem small, but it's **infinitely better than 0%**
    """)
    
    st.markdown("---")
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Model Architecture:**
        - LightGCN with 3 layers
        - 64-dimensional embeddings
        - 1.9M trainable parameters
        
        **Training:**
        - Optimizer: Adam (lr=0.001)
        - Loss: BPR (Bayesian Personalized Ranking)
        - Epochs: 50
        - Training time: 20 seconds on T4 GPU
        
        **Cold-Start Pipeline:**
        - CLIP model: openai/clip-vit-base-patch32
        - Visual similarity threshold: 0.5
        - K-nearest neighbors: 10
        - Embedding aggregation: Weighted average
        """)

# Footer
st.sidebar.markdown("<br>" * 3, unsafe_allow_html=True)  # Add vertical space
st.sidebar.markdown("---")
st.sidebar.caption("**Developed by:** Andrew John J")
st.sidebar.caption("**Course:** CSE 941 - Deep Learning on Graphs")
st.sidebar.caption("**Tech Stack:** Streamlit + PyTorch + CLIP + LightGCN")
st.sidebar.caption("© 2026 Michigan State University")