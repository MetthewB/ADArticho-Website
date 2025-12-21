import numpy as np

def show_category_examples(dataset, dataset_name, topics_list, score_order = False, n=10):
    """
    Prints for each topic the best or worst matched captions by topic
    Args: 
        dataset (pd.Dataframe): dataframe containing the captions, their topic and matching scores.
        dataset_name (str): Name of the dataset
        topic_list (str): list of topics
        score_order (bool): True to show bottom ranking and False to show top ranking
        n (int): Number of caption to print per topic
    Return:
        None
        
    """
    if score_order:    
        print(f"{dataset_name}'s {n}-worst matched captions by topic and their matching score:\n")
    else:
        print(f"{dataset_name}'s {n}-best matched captions by topic and their matching score:\n")
    for topic in topics_list:
        print(f"\n=== {topic} category Examples ===")
        samples = dataset[dataset["Topic"] == topic].sort_values('matching_score', ascending=score_order).head(n)[['caption','matching_score']]
        for i in range(len(samples)):
            print(f"- {samples.iloc[i]['caption']} --- {samples.iloc[i]['matching_score']}")