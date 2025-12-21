import pandas as pd
from src.data.load_data import NewYorkerDataset, OxfordDataset
from src.data.preprocess import preprocess_caption
from src.data.preprocess import preprocess_caption_lr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def load_datasets():
    newyork_path = "data/newyorker_caption_contest/data"
    oxford_path = "data/oxford_hic"
    newyorker_data = NewYorkerDataset(newyork_path)
    oxford_data = OxfordDataset(oxford_path)
    return newyorker_data, oxford_data

def get_combined_dataframe(newyorker_data, oxford_data):
    df_ny = pd.DataFrame(newyorker_data.data["caption"].copy())
    df_ny["label"] = 0   # New Yorker

    df_hic = pd.DataFrame(oxford_data.data["caption"].copy())
    df_hic["label"] = 1   # HIC

    df = pd.concat([df_ny, df_hic], ignore_index=True)
    return df

def preprocess_dataframe(df):
    df_copy = df.copy()
    df_copy["caption"] = df_copy["caption"].apply(preprocess_caption_lr)
    df_copy = df_copy[df_copy["caption"].str.strip() != ""].reset_index(drop=True)
    return df_copy

def vectorize_captions(df, max_features=30000, ngram_range=(1,2)):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(df["caption"])
    y = df["label"].values
    return X, y, vectorizer

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=500, solver='lbfgs', class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def get_top_features(model, vectorizer, top_n=50):
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_positive_indices = coefs.argsort()[-top_n:][::-1]
    top_negative_indices = coefs.argsort()[:top_n]

    top_positive_features = {feature_names[i]: coefs[i] for i in top_positive_indices}
    top_negative_features = {feature_names[i]: coefs[i] for i in top_negative_indices}

    print("Top positive tokens (Oxford HIC):")
    for feature, coef in top_positive_features.items():
        print(f"{feature}: {coef:.4f}")
    print("\nTop negative tokens (New Yorker):")
    for feature, coef in top_negative_features.items():
        print(f"{feature}: {coef:.4f}")
    return top_positive_features, top_negative_features

def generate_wordcloud(features, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(features)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()
    
def main(top_n=50):
    print("Loading datasets...")
    newyorker_data, oxford_data = load_datasets()
    print("Preprocessing datasets...")
    df = get_combined_dataframe(newyorker_data, oxford_data)
    df_preprocessed = preprocess_dataframe(df)
    print("Vectorizing captions...")
    X, y, vectorizer = vectorize_captions(df_preprocessed)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Training Logistic Regression model...")
    model = train_logistic_regression(X_train, y_train)
    print("Evaluating model...")
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Extracting top features...")
    top_positive_features, top_negative_features = get_top_features(model, vectorizer, top_n=top_n)
    print("Generating word clouds...")
    generate_wordcloud(top_positive_features, "Top Tokens Oxford HIC")

    generate_wordcloud(top_negative_features, "Top Tokens New Yorker")

