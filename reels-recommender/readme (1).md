# Instagram Reels/Shorts Recommender System

This is a **content-based + hybrid recommender system** for Instagram Reels / YouTube Shorts.  
The system suggests relevant short videos to users based on:
- Post content (TF-IDF, Word2Vec embeddings)
- User interactions (likes, comments, replies as proxy for watch-time)
- Hybrid scoring to improve recommendations

## 🚀 Features
- TF-IDF and Word2Vec embeddings for posts
- User profile building based on engagement
- Hybrid recommendation (content + interaction)
- Streamlit web app for easy demo

## 📂 Dataset
The dataset comes from a **Kaggle Instagram Reels dataset**, preprocessed for recommendation tasks.  

## 🛠 Tech Stack
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Gensim
- NLTK

## ▶️ Run Locally
```bash
streamlit run app.py
