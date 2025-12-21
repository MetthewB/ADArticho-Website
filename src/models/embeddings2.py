import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def build_category_embeddings(categories, model):
    """
    Build embeddings for each category description.
    
    Args:
        categories (dict): Dictionary of topics and their descriptive words.
        model: The sentence embedding model.
    Return:
        cat_names (list): List of topics to be classified as.
        cat_emb (np.array): Matrix of embeddings of each topic.
    """
    cat_names = list(categories.keys())
    cat_emb = np.vstack([model.encode(categories[c]) for c in cat_names])
    return cat_names, cat_emb


def classify_captions(dataset, model, cat_names, cat_matrix, batch_size=256):
    """
    Classifies captions by computing embeddings batch-by-batch,
    comparing to category embeddings, and storing only topic and score.
    
    Args:
        dataset (pd.DataFrame): The dataset containing captions.
        model: The sentence embedding model.
        cat_names (list): List of topics to be classified as.
        cat_matrix (np.array): Matrix of embeddings of each topic.
        batch_size (int): The number of batches to classify.
        
    Return:
        pd.DataFrame: The classified dataset with topics and matching score.
    """
    data = dataset.copy()
    sentences = data["caption"].tolist()

    topic_category = []
    topic_score = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Classifying captions"):
        batch = sentences[i:i+batch_size]

        # Compute embeddings (not stored)
        emb_batch = model.encode(batch, show_progress_bar=False)

        # Compute similarities
        sim_matrix = cosine_similarity(emb_batch, cat_matrix)

        # Best category index per caption
        idxs = np.argmax(sim_matrix, axis=1)

        topic_category.extend([cat_names[j] for j in idxs])
        topic_score.extend(sim_matrix[np.arange(len(idxs)), idxs])

    # Store outputs only
    data["Topic"] = topic_category
    data["matching_score"] = topic_score

    return data


def classify_captions_sample_embeddings(dataset, model, categories, batch_size=256):
    """
    Full classification pipeline combining:
    - category embedding generation
    - batch classification
    Args:
        dataset (pd.DataFrame): The dataset containing captions.
        model: The sentence embedding model.
        categories (dict): Dictionary of topics and their descriptive words.
        batch_size (int): The number of batches to classify.
        
    Return:
        pd.DataFrame: The classified dataset with topics and matching score.
    """
    cat_names, cat_matrix = build_category_embeddings(categories, model)
    return classify_captions(dataset, model, cat_names, cat_matrix, batch_size)

