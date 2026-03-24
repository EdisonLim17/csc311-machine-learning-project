import json
import numpy as np
import pandas as pd
import re

# --- 1. Loading Functions ---
def load_model_assets(asset_path="."):
    """
    Loads all necessary files for prediction.
    This includes model weights, preprocessing stats, and vocabularies.
    """
    with open(f"{asset_path}/model_params.json", 'r') as f:
        params = json.load(f)

    model_weights = np.array(params['weights'])
    model_intercept = np.array(params['intercept'])
    
    with open(f"{asset_path}/preprocessing_stats.json", 'r') as f:
        stats = json.load(f)

    with open(f"{asset_path}/tfidf_vocab.json", 'r') as f:
        tfidf_vocab = json.load(f)

    return {
        "weights": model_weights,
        "intercept": model_intercept,
        "imputation_means": stats["imputation_means"],
        "imputation_modes": stats["imputation_modes"],
        "numerical_features": stats["numerical_features"],
        "categorical_features": stats["categorical_features"],
        "text_features": stats["text_features"],
        "one_hot_categories": stats["one_hot_categories"],
        "feature_means": np.array(stats["feature_means"]),
        "feature_stds": np.array(stats["feature_stds"]),
        "tfidf_vocab": tfidf_vocab,
        "tfidf_idf": np.array(stats["tfidf_idf"])
    }

# --- 2. Preprocessing Functions ---
def preprocess_text(text):
    """A simple text cleaner."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text) # Remove non-alphanumeric
    return text

def tfidf_transform(texts, vocab, idf):
    """Transforms text into TF-IDF vectors using a pre-built vocab and IDF scores."""
    term_to_idx = {term: i for i, term in enumerate(vocab)}
    n_features = len(vocab)
    tf = np.zeros((len(texts), n_features), dtype=float)

    for i, text in enumerate(texts):
        cleaned_text = preprocess_text(text)
        words = cleaned_text.split()
        term_counts = {}
        for word in words:
            if word in term_to_idx:
                term_counts[word] = term_counts.get(word, 0) + 1
        
        if len(words) > 0:
            for word, count in term_counts.items():
                idx = term_to_idx[word]
                tf[i, idx] = count / len(words)

    return tf * idf

def preprocess_data(df, assets):
    """
    Applies all preprocessing steps to the input DataFrame.
    """
    # Combine text features first, same as in training
    df['combined_text'] = df[assets["text_features"]].fillna('').agg(' '.join, axis=1)

    # Impute missing values
    for col in assets["numerical_features"]:
        # Coerce to numeric, as test data might also have bad values
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(assets["imputation_means"][col], inplace=True)
    for col in assets["categorical_features"]:
        df[col].fillna(assets["imputation_modes"][col], inplace=True)

    # One-Hot Encode categorical features
    ohe_features = []
    for col, categories in assets["one_hot_categories"].items():
        for category in categories:
            feature_name = f"{col}_{category}"
            ohe_features.append(pd.Series(df[col] == category, name=feature_name, dtype=int))
    ohe_df = pd.concat(ohe_features, axis=1)

    # TF-IDF for text features
    text_col = 'combined_text'
    tfidf_vectors = tfidf_transform(df[text_col], assets["tfidf_vocab"], assets["tfidf_idf"])
    tfidf_df = pd.DataFrame(tfidf_vectors, columns=[f"tfidf_{i}" for i in range(tfidf_vectors.shape[1])])

    # Combine features
    numerical_df = df[assets["numerical_features"]]
    X_combined = pd.concat([numerical_df.reset_index(drop=True),
                            ohe_df.reset_index(drop=True),
                            tfidf_df.reset_index(drop=True)], axis=1)

    # Standardize features
    # Ensure columns are in the same order as during training
    # This is handled implicitly by the order of concatenation
    X_standardized = (X_combined.to_numpy() - assets["feature_means"]) / assets["feature_stds"]
    
    # Handle potential division by zero if a feature had zero std dev
    X_standardized = np.nan_to_num(X_standardized)

    return X_standardized


# --- 3. Prediction Functions ---
def softmax(z):
    """Softmax function to compute probabilities."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def predict(X, weights, intercept):
    """Makes predictions using a logistic regression model."""
    logits = X @ weights.T + intercept
    probabilities = softmax(logits)
    return np.argmax(probabilities, axis=1)


# --- 4. Main Entry Point ---
def predict_all(csv_file_path: str):
    """
    Main function to load data, preprocess, and predict.
    
    Args:
        csv_file_path (str): Path to the input CSV file.
        
    Returns:
        np.ndarray: An array of predicted class labels.
    """
    # Load assets
    assets = load_model_assets()
    
    # Load and preprocess data
    df = pd.read_csv(csv_file_path)
    X_processed = preprocess_data(df.copy(), assets)
    
    # Make predictions
    predictions = predict(X_processed, assets["weights"], assets["intercept"])
    
    return predictions

# Example usage (for testing purposes)
if __name__ == '__main__':
    # This block will not be run by the grader but is useful for your own testing.
    # You would need to create a dummy 'test.csv' and the asset files.
    # For example:
    # predictions = predict_all('path/to/your/test_data.csv')
    # print(predictions)
    pass