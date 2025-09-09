import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from scipy.sparse import hstack, issparse
import warnings

warnings.filterwarnings('ignore')

# Download NLTK tokenizer
try:
    nltk.download('punkt', quiet=True)
except:
    st.warning("NLTK punkt tokenizer download failed. Text processing may be affected.")

# Configuration with defaults
DEFAULT_CONFIG = {
    "USE_PRETRAINED": False,  # Avoid kagglehub dependency
    "TFIDF_WEIGHT": 0.7,
    "W2V_WEIGHT": 0.3,
    "LIKE_WEIGHT": 0.005,  # Reduced to avoid high scores
    "REPLY_WEIGHT": 0.005, # Reduced to avoid high scores
    "TOP_N": 5
}

# -------------------------
# Utility Functions
# -------------------------
def clean_text(text):
    """Clean and preprocess text data."""
    if pd.isna(text) or text == "":
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)        # Remove URLs
    text = re.sub(r"[@#]\S+", " ", text)        # Remove mentions/hashtags
    text = re.sub(r"[^a-z0-9\s]", " ", text)    # Keep alphanumerics
    text = re.sub(r"\s+", " ", text).strip()
    return text

def safe_tokenize(text):
    """Safely tokenize text with fallback."""
    try:
        return word_tokenize(text.lower())
    except:
        return text.lower().split()

# -------------------------
# Load and Clean Dataset
# -------------------------
@st.cache_data
def load_data(_uploaded_file=None):
    """Load dataset from file upload or local file."""
    try:
        if _uploaded_file is not None:
            df = pd.read_csv(_uploaded_file)
        else:
            df = pd.read_csv("Instagram-datasets.csv")
        
        required_cols = ["comment", "hashtag_comment", "tagged_users_in_comment", 
                        "post_id", "post_user", "post_url", "likes_number", 
                        "replies_number", "comment_user"]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
            
        for col in ["comment", "hashtag_comment", "tagged_users_in_comment"]:
            df[col] = df[col].fillna("")
        
        df["likes_number"] = pd.to_numeric(df["likes_number"], errors='coerce').fillna(0)
        df["replies_number"] = pd.to_numeric(df["replies_number"], errors='coerce').fillna(0)
        
        st.success(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        st.error("Instagram-datasets.csv not found. Please upload the file using the file uploader.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# -------------------------
# Construct Post Documents
# -------------------------
@st.cache_data
def construct_post_docs(_df):
    """Construct post documents by aggregating comments."""
    if _df is None or _df.empty:
        st.error("Cannot construct post documents: Dataset is not loaded.")
        return None
    
    try:
        text_parts = ["comment", "hashtag_comment", "tagged_users_in_comment"]
        _df["__doc_part__"] = _df[text_parts].apply(
            lambda x: clean_text(" ".join(x.astype(str))), axis=1
        )
        
        post_docs = (
            _df.groupby("post_id")
            .agg({
                "__doc_part__": lambda x: " ".join(x.unique()),
                "post_user": "first",
                "post_url": "first",
                "likes_number": "sum",
                "replies_number": "sum"
            })
            .reset_index()
            .rename(columns={"__doc_part__": "post_text"})
        )
        
        post_docs = post_docs[post_docs["post_text"].str.strip() != ""]
        
        st.success(f"Post documents constructed. Shape: {post_docs.shape}")
        return post_docs
        
    except Exception as e:
        st.error(f"Error constructing post documents: {str(e)}")
        return None

# -------------------------
# Build TF-IDF Embeddings
# -------------------------
@st.cache_resource
def build_tfidf(_post_docs):
    """Build TF-IDF matrix from post documents."""
    if _post_docs is None or _post_docs.empty:
        st.error("Cannot build TF-IDF: Post documents are not available.")
        return None, None
    
    try:
        tfidf = TfidfVectorizer(
            max_features=5000, 
            stop_words="english", 
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        tfidf_matrix = tfidf.fit_transform(_post_docs["post_text"])
        st.success(f"TF-IDF matrix built. Shape: {tfidf_matrix.shape}")
        return tfidf, tfidf_matrix
        
    except Exception as e:
        st.error(f"Error building TF-IDF: {str(e)}")
        return None, None

# -------------------------
# Build Word2Vec Embeddings
# -------------------------
@st.cache_resource
def build_w2v(_post_docs, use_pretrained=False):
    """Build Word2Vec embeddings."""
    if _post_docs is None or _post_docs.empty:
        st.error("Cannot build Word2Vec: Post documents are not available.")
        return None, None, None
    
    try:
        sentences = [safe_tokenize(text) for text in _post_docs["post_text"]]
        sentences = [sent for sent in sentences if len(sent) > 0]
        
        def get_w2v_vector(tokens, model, size):
            vec = np.zeros(size)
            count = 0
            for word in tokens:
                try:
                    if hasattr(model, 'wv') and word in model.wv:
                        vec += model.wv[word]
                        count += 1
                except KeyError:
                    continue
            return vec / count if count > 0 else vec
        
        if use_pretrained:
            st.warning("Pretrained Word2Vec not supported in this version. Using trained model.")
            use_pretrained = False
        
        w2v_model = Word2Vec(
            sentences, 
            vector_size=100, 
            window=5, 
            min_count=1, 
            workers=4,
            epochs=10
        )
        vector_size = 100
        
        w2v_vectors = np.array([
            get_w2v_vector(safe_tokenize(text), w2v_model, vector_size) 
            for text in _post_docs["post_text"]
        ])
        
        st.success(f"Word2Vec vectors built. Shape: {w2v_vectors.shape}")
        return w2v_model, w2v_vectors, vector_size
        
    except Exception as e:
        st.error(f"Error building Word2Vec: {str(e)}")
        return None, None, None

# -------------------------
# Build Hybrid Matrix
# -------------------------
@st.cache_resource
def build_hybrid_matrix(_tfidf_matrix, _w2v_vectors):
    """Combine TF-IDF and Word2Vec into hybrid matrix."""
    if _tfidf_matrix is None or _w2v_vectors is None:
        st.error("Cannot build hybrid matrix: TF-IDF or Word2Vec vectors are missing.")
        return None
    
    try:
        if _tfidf_matrix.shape[0] != _w2v_vectors.shape[0]:
            st.error(f"Shape mismatch: TF-IDF {_tfidf_matrix.shape} vs W2V {_w2v_vectors.shape}")
            return None
        
        hybrid_matrix = hstack([_tfidf_matrix, _w2v_vectors])
        st.success(f"Hybrid matrix built. Shape: {hybrid_matrix.shape}")
        return hybrid_matrix
        
    except Exception as e:
        st.error(f"Error building hybrid matrix: {str(e)}")
        return None

# -------------------------
# Compute Engagement
# -------------------------
@st.cache_data
def compute_engagement(_df):
    """Compute engagement scores for user-post interactions."""
    if _df is None or _df.empty:
        st.error("Cannot compute engagement: Dataset is not loaded.")
        return None
    
    try:
        _df = _df.copy()
        _df["__engagement__"] = (
            1.0 + 
            0.1 * _df["likes_number"].astype(float) + 
            0.2 * _df["replies_number"].astype(float)
        )
        
        user_post_eng = (
            _df.groupby(["comment_user", "post_id"])["__engagement__"]
            .sum()
            .reset_index()
        )
        
        st.success(f"Engagement computed. Shape: {user_post_eng.shape}")
        return user_post_eng
        
    except Exception as e:
        st.error(f"Error computing engagement: {str(e)}")
        return None

# -------------------------
# Recommendation Functions
# -------------------------
def make_user_vector(user_name, user_post_eng, post_docs, hybrid_matrix, 
                    tfidf_weight, w2v_weight, tfidf_matrix):
    """Create user preference vector from interactions."""
    if any(x is None for x in [user_post_eng, post_docs, hybrid_matrix]):
        return None, None
    
    interactions = user_post_eng[user_post_eng["comment_user"] == user_name]
    if interactions.empty:
        return None, None
    
    postid_to_idx = {pid: i for i, pid in enumerate(post_docs["post_id"])}
    
    valid_interactions = interactions[interactions["post_id"].isin(postid_to_idx.keys())]
    if valid_interactions.empty:
        return None, None
    
    idxs = [postid_to_idx[pid] for pid in valid_interactions["post_id"]]
    weights = valid_interactions["__engagement__"].values.astype(float)
    
    if len(weights) == 0 or weights.sum() == 0:
        weights = np.ones(len(idxs))
    
    if issparse(hybrid_matrix):
        selected_vectors = np.vstack([hybrid_matrix.getrow(i).toarray() for i in idxs])
    else:
        selected_vectors = hybrid_matrix[idxs]
    
    tfidf_size = tfidf_matrix.shape[1] if tfidf_matrix is not None else 0
    selected_vectors = selected_vectors.copy()
    selected_vectors[:, :tfidf_size] *= tfidf_weight
    selected_vectors[:, tfidf_size:] *= w2v_weight
    
    weights = weights / weights.sum()
    user_vec = np.average(selected_vectors, axis=0, weights=weights)
    
    return user_vec.reshape(1, -1), set(valid_interactions["post_id"])

def recommend_for_user(user_name, k, exclude_seen, engagement_boost, 
                      like_weight, reply_weight, tfidf_weight, w2v_weight,
                      user_post_eng, post_docs, hybrid_matrix, tfidf_matrix):
    """Generate recommendations for a user."""
    try:
        if any(x is None for x in [user_post_eng, post_docs, hybrid_matrix]):
            return pd.DataFrame()
        
        user_vec, seen_posts = make_user_vector(
            user_name, user_post_eng, post_docs, hybrid_matrix,
            tfidf_weight, w2v_weight, tfidf_matrix
        )
        
        if user_vec is None:
            if user_post_eng is not None:
                global_scores = (
                    user_post_eng.groupby("post_id")["__engagement__"]
                    .sum()
                    .reset_index()
                    .sort_values("__engagement__", ascending=False)
                )
                top_posts = global_scores.head(k)
                result = post_docs[post_docs["post_id"].isin(top_posts["post_id"])].copy()
                result = result.merge(top_posts, on="post_id", how="left")
                result["score"] = result["__engagement__"]
                return result[["post_id", "score", "post_user", "post_url", "post_text", "likes_number", "replies_number"]]
            else:
                return post_docs.head(k)[["post_id", "post_user", "post_url", "post_text", "likes_number", "replies_number"]].assign(score=1.0)
        
        if issparse(hybrid_matrix):
            sim_scores = cosine_similarity(user_vec, hybrid_matrix.toarray()).ravel()
        else:
            sim_scores = cosine_similarity(user_vec, hybrid_matrix).ravel()
        
        rec_df = post_docs.copy()
        rec_df["score"] = sim_scores
        
        if engagement_boost:
            rec_df["score"] = rec_df["score"] * (
                1 + like_weight * rec_df["likes_number"] + 
                reply_weight * rec_df["replies_number"]
            )
        
        if exclude_seen and seen_posts:
            rec_df = rec_df[~rec_df["post_id"].isin(seen_posts)]
        
        rec_df = rec_df.sort_values("score", ascending=False).head(k)
        return rec_df[["post_id", "score", "post_user", "post_url", "post_text", "likes_number", "replies_number"]]
    
    except Exception as e:
        st.error(f"Error in recommend_for_user: {str(e)}")
        return pd.DataFrame()

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(
        page_title="Instagram Reels Recommender",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Instagram Reels Hybrid Recommender")
    st.markdown("This app recommends reels using a **TF-IDF + Word2Vec** hybrid model based on user interactions or post similarity.")
    
    # File upload
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Instagram Dataset (CSV)", 
        type=['csv'],
        help="Upload your Instagram-datasets.csv file"
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is None:
        st.warning("Please upload a valid Instagram dataset CSV file to continue.")
        st.info("The CSV should contain columns: comment, hashtag_comment, tagged_users_in_comment, post_id, post_user, post_url, likes_number, replies_number, comment_user")
        return
    
    # Sidebar parameters
    st.sidebar.header("🔧 Model Parameters")
    tfidf_weight = st.sidebar.slider("TF-IDF Weight", 0.0, 1.0, DEFAULT_CONFIG["TFIDF_WEIGHT"], 0.1)
    w2v_weight = st.sidebar.slider("Word2Vec Weight", 0.0, 1.0, DEFAULT_CONFIG["W2V_WEIGHT"], 0.1)
    like_weight = st.sidebar.slider("Like Weight", 0.0, 0.1, DEFAULT_CONFIG["LIKE_WEIGHT"], 0.001)
    reply_weight = st.sidebar.slider("Reply Weight", 0.0, 0.1, DEFAULT_CONFIG["REPLY_WEIGHT"], 0.001)
    top_n = st.sidebar.slider("Number of Recommendations", 1, 20, DEFAULT_CONFIG["TOP_N"])
    engagement_boost = st.sidebar.checkbox("Apply Engagement Boost", value=True)
    exclude_seen = st.sidebar.checkbox("Exclude Seen Posts (User-Based)", value=True)
    
    # Initialize components
    with st.spinner("🔄 Processing data..."):
        progress_bar = st.progress(0)
        
        progress_bar.progress(20)
        post_docs = construct_post_docs(df)
        
        if post_docs is None:
            st.error("Failed to construct post documents.")
            return
        
        progress_bar.progress(40)
        tfidf, tfidf_matrix = build_tfidf(post_docs)
        
        progress_bar.progress(60)
        w2v_model, w2v_vectors, vector_size = build_w2v(post_docs, DEFAULT_CONFIG["USE_PRETRAINED"])
        
        progress_bar.progress(80)
        hybrid_matrix = build_hybrid_matrix(tfidf_matrix, w2v_vectors)
        
        progress_bar.progress(90)
        user_post_eng = compute_engagement(df)
        
        progress_bar.progress(100)
        
    if any(x is None for x in [post_docs, tfidf_matrix, hybrid_matrix, user_post_eng]):
        st.error("Failed to initialize recommendation system. Please check your data and try again.")
        return
    
    st.success("✅ Recommendation system initialized successfully!")
    
    # Mode selection
    col1, col2 = st.columns([1, 3])
    with col1:
        recommendation_mode = st.selectbox(
            "Choose Recommendation Mode:",
            ["User-Based", "Post-Based"],
            help="User-Based: Personalized recommendations\nPost-Based: Similar posts"
        )
    
    if recommendation_mode == "User-Based":
        st.subheader("👤 User-Based Recommendations")
        
        user_options = sorted(user_post_eng["comment_user"].unique())
        
        if not user_options:
            st.warning("No users found in the dataset.")
            return
            
        selected_user = st.selectbox(
            "Select a User:",
            user_options,
            help="Choose a user to get personalized recommendations"
        )
        
        if selected_user and st.button("🎯 Generate Recommendations", type="primary"):
            with st.spinner("Generating personalized recommendations..."):
                recs = recommend_for_user(
                    selected_user, k=top_n, exclude_seen=exclude_seen,
                    engagement_boost=engagement_boost, like_weight=like_weight,
                    reply_weight=reply_weight, tfidf_weight=tfidf_weight,
                    w2v_weight=w2v_weight, user_post_eng=user_post_eng,
                    post_docs=post_docs, hybrid_matrix=hybrid_matrix,
                    tfidf_matrix=tfidf_matrix
                )
            
            if not recs.empty:
                st.subheader(f"🎥 Top Recommendations for {selected_user}")
                
                display_recs = recs.copy()
                display_recs["score"] = display_recs["score"].round(3)
                display_recs["post_text_short"] = display_recs["post_text"].str[:100] + "..."
                
                st.dataframe(
                    display_recs[["post_id", "score", "post_user", "post_text_short", "likes_number", "replies_number"]],
                    use_container_width=True,
                    column_config={
                        "post_id": "Post ID",
                        "score": st.column_config.NumberColumn("Score", format="%.3f"),
                        "post_user": "Creator",
                        "post_text_short": "Content Preview",
                        "likes_number": st.column_config.NumberColumn("Likes", format="%d"),
                        "replies_number": st.column_config.NumberColumn("Replies", format="%d")
                    }
                )
                
                # Download button
                csv = display_recs.to_csv(index=False)
                st.download_button(
                    "📥 Download Recommendations",
                    csv,
                    f"recommendations_{selected_user}.csv",
                    "text/csv"
                )
                
                with st.expander(f"📊 {selected_user}'s Interaction History"):
                    user_interactions = user_post_eng[user_post_eng["comment_user"] == selected_user]
                    if not user_interactions.empty:
                        interaction_details = user_interactions.merge(
                            post_docs[["post_id", "post_text", "post_user"]], 
                            on="post_id", 
                            how="left"
                        )
                        interaction_details["post_text_short"] = interaction_details["post_text"].str[:80] + "..."
                        st.dataframe(
                            interaction_details[["post_id", "post_user", "post_text_short", "__engagement__"]],
                            use_container_width=True,
                            column_config={
                                "post_id": "Post ID",
                                "post_user": "Creator",
                                "post_text_short": "Content",
                                "__engagement__": st.column_config.NumberColumn("Engagement", format="%.2f")
                            }
                        )
                    else:
                        st.info("No interaction history found.")
            else:
                st.warning("No recommendations could be generated for this user.")
    
    else:
        st.subheader("📱 Post-Based Recommendations")
        
        post_options = []
        post_mapping = {}
        for idx, row in post_docs.iterrows():
            display_text = f"ID: {row['post_id']} | {row['post_user']} | {row['post_text'][:60]}..."
            post_options.append(display_text)
            post_mapping[display_text] = row['post_id']
        
        selected_post_display = st.selectbox(
            "Select a Post:",
            post_options,
            help="Choose a post to find similar content"
        )
        
        if selected_post_display and st.button("🔍 Find Similar Posts", type="primary"):
            selected_post_id = post_mapping[selected_post_display]
            
            postid_to_idx = {pid: i for i, pid in enumerate(post_docs["post_id"])}
            
            if selected_post_id not in postid_to_idx:
                st.error("Selected post not found in processed data.")
                return
            
            index = postid_to_idx[selected_post_id]
            
            st.subheader("📌 Selected Post")
            selected_post_data = post_docs[post_docs["post_id"] == selected_post_id].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Post ID", selected_post_data["post_id"])
            with col2:
                st.metric("Likes", int(selected_post_data["likes_number"]))
            with col3:
                st.metric("Replies", int(selected_post_data["replies_number"]))
            
            st.text_area("Content:", selected_post_data["post_text"], height=100, disabled=True)
            
            with st.spinner("Finding similar posts..."):
                if issparse(hybrid_matrix):
                    post_vec = hybrid_matrix.getrow(index).toarray()
                else:
                    post_vec = hybrid_matrix[index:index+1]
                
                if issparse(hybrid_matrix):
                    sim_scores = cosine_similarity(post_vec, hybrid_matrix.toarray()).ravel()
                else:
                    sim_scores = cosine_similarity(post_vec, hybrid_matrix).ravel()
                
                similar_indices = sim_scores.argsort()[::-1][1:top_n+1]
                
                recs = post_docs.iloc[similar_indices].copy()
                recs["score"] = sim_scores[similar_indices]
                
                if engagement_boost and (like_weight > 0 or reply_weight > 0):
                    recs["score"] = recs["score"] * (
                        1 + like_weight * recs["likes_number"] + 
                        reply_weight * recs["replies_number"]
                    )
                
                recs = recs.sort_values("score", ascending=False)
            
            st.subheader("🎯 Similar Posts")
            if not recs.empty:
                display_recs = recs.copy()
                display_recs["score"] = display_recs["score"].round(3)
                display_recs["post_text_short"] = display_recs["post_text"].str[:100] + "..."
                
                st.dataframe(
                    display_recs[["post_id", "score", "post_user", "post_text_short", "likes_number", "replies_number"]],
                    use_container_width=True,
                    column_config={
                        "post_id": "Post ID",
                        "score": st.column_config.NumberColumn("Similarity", format="%.3f"),
                        "post_user": "Creator",
                        "post_text_short": "Content Preview",
                        "likes_number": st.column_config.NumberColumn("Likes", format="%d"),
                        "replies_number": st.column_config.NumberColumn("Replies", format="%d")
                    }
                )
                
                csv = display_recs.to_csv(index=False)
                st.download_button(
                    "📥 Download Similar Posts",
                    csv,
                    f"similar_posts_{selected_post_id}.csv",
                    "text/csv"
                )
            else:
                st.warning("No similar posts found.")

if __name__ == "__main__":
    main()
