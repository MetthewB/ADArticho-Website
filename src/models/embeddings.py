from sklearn.metrics.pairwise import cosine_similarity


def generate_caption_embeddings(dataset, model):
    """
    Generates embeddings for captions in the dataset.

    Args:
        dataset (pd.DataFrame): The dataset containing captions.
        model: The sentence embedding model.

    Returns:
        pd.DataFrame: The dataset with an additional 'embeddings' column.
    """
    sentences = dataset["caption"].tolist()
    embeddings = model.encode(sentences)
    dataset["embeddings"] = embeddings.tolist()
    return dataset


def generate_category_embeddings(humor_categories, model):
    """
    Generates embeddings for humor categories.

    Args:
        humor_categories (dict): A dictionary of humor categories with descriptions.
        model: The sentence embedding model.

    Returns:
        dict: A dictionary mapping humor categories to their embeddings.
    """
    return {
        category: model.encode(description)
        for category, description in humor_categories.items()
    }


def classify_captions_with_embeddings(dataset, category_embeddings):
    """
    Classifies captions based on their similarity to humor category embeddings.

    Args:
        dataset (pd.DataFrame): The dataset containing caption embeddings.
        category_embeddings (dict): A dictionary of humor category embeddings.

    Returns:
        pd.DataFrame: The dataset with an additional 'humor_type' column.
    """
    humor_types = []
    for caption_embedding in dataset["embeddings"]:
        # Calculate cosine similarity between the caption and each humor category
        similarities = {
            category: cosine_similarity([caption_embedding], [embedding])[0][0]
            for category, embedding in category_embeddings.items()
        }
        # Find the category with the highest similarity
        best_category = max(similarities, key=similarities.get)
        humor_types.append(best_category)
    dataset["humor_type"] = humor_types
    return dataset


def classify_captions_sample_embeddings(
    dataset, sample_size, model, humor_categories, threshold=0.3
):
    """
    Processes and classifies captions using embeddings.

    Args:
        dataset (pd.DataFrame): The dataset containing captions.
        sample_size (int): The number of samples to classify.
        humor_categories (dict): A dictionary of humor categories with descriptions.
        model: The sentence embedding model.
        threshold (float): Minimum similarity score for classification (default: 0.3)

    Returns:
        pd.DataFrame: The classified dataset with humor types.
    """
    # Step 1: Take a random sample of the dataset
    sampled_data = dataset.sample(n=sample_size, random_state=42).copy()

    # Step 2: Generate embeddings for the sampled captions
    sampled_data = generate_caption_embeddings(sampled_data, model)

    # Step 3: Generate humor category embeddings
    category_embeddings = generate_category_embeddings(humor_categories, model)

    # Step 4: Classify captions based on embeddings
    classified_data = classify_captions_with_embeddings(
        sampled_data, category_embeddings, threshold=threshold
    )

    return classified_data
