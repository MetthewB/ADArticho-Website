from src.data.preprocess import preprocess_caption

def classify_humor(caption, classifier, humor_categories):
    """
    Classifies the humor type of a given caption using a zero-shot classification model.

    Args:
        caption (str): The caption to classify.
        classifier: The zero-shot classification pipeline.
        humor_categories (dict): A dictionary of humor categories with descriptions.

    Returns:
        str: The predicted humor type for the caption.
    """
    # Preprocess the caption
    caption = preprocess_caption(caption)
    # Perform zero-shot classification on the caption
    result = classifier(caption, candidate_labels=list(humor_categories.keys()))
    # Return the top predicted humor type
    return result["labels"][0]

def classify_captions(dataset, classifier, humor_categories):
    """
    Classifies the humor type for all captions in a dataset.

    Args:
        dataset (pd.DataFrame): The dataset containing captions to classify.
        classifier: The zero-shot classification pipeline.
        humor_categories (dict): A dictionary of humor categories with descriptions.

    Returns:
        pd.DataFrame: The dataset with an additional 'humor_type' column containing predictions.
    """
    # Preprocess captions and classify humor type
    dataset["caption"] = dataset["caption"].apply(preprocess_caption)
    dataset["humor_type"] = dataset["caption"].apply(lambda x: classify_humor(x, classifier, humor_categories))
    return dataset

def classify_captions_sample(dataset, sample_size, classifier, humor_categories):
    """
    Classifies the humor type for a random sample of captions in a dataset.

    Args:
        dataset (pd.DataFrame): The dataset containing captions to classify.
        sample_size (int): The number of samples to classify.
        classifier: The zero-shot classification pipeline.
        humor_categories (dict): A dictionary of humor categories with descriptions.

    Returns:
        pd.DataFrame: The sampled dataset with an additional 'humor_type' column containing predictions.
    """
    # Take a random sample of the dataset
    sample = dataset.sample(n=sample_size, random_state=42).copy()
    # Preprocess captions and classify humor type
    sample["caption"] = sample["caption"].apply(preprocess_caption)
    sample["humor_type"] = sample["caption"].apply(lambda x: classify_humor(x, classifier, humor_categories))
    return sample