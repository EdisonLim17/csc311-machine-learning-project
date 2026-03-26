import pandas as pd
import pickle
import re
import math

# --- Load all necessary artifacts ---
try:
    with open('rf_manual_model.pkl', 'rb') as f:
        forest_structure = pickle.load(f)
    with open('preprocessor_components.pkl', 'rb') as f:
        preprocessor_components = pickle.load(f)
    with open('final_label_encoder.pkl', 'rb') as f:
        label_encoder_classes = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: A required .pkl file is missing: {e}")
    exit()

# --- Manual Preprocessing Functions ---

def manual_impute_and_scale(df, params):
    """
    Manually imputes and scales MULTIPLE numerical features.
    """
    imputer_means = params['imputer_mean_']
    scaler_means = params['mean_']
    scaler_scales = params['scale_']
    
    # Create a copy to avoid modifying the original DataFrame slice
    df_copy = df.copy()
    
    # Impute and scale each column
    for i, col in enumerate(df_copy.columns):
        df_copy[col] = df_copy[col].fillna(imputer_means[i])
        df_copy[col] = (df_copy[col] - scaler_means[i]) / scaler_scales[i]
        
    return df_copy.values.tolist()

def manual_one_hot_encode(df, params):
    """
    Manually performs one-hot encoding, correctly handling unknown categories
    to mimic handle_unknown='ignore'.
    """
    encoded_matrix = []
    all_categories = params['categories_']
    
    # These are the categories the model was trained on.
    room_cats = all_categories[0]
    season_cats = all_categories[1]
    
    for index, row in df.iterrows():
        encoded_row = []
        
        # Encode 'room'
        for cat in room_cats:
            # The value is 1 only if the row's value matches a known category.
            encoded_row.append(1.0 if row['room'] == cat else 0.0)
            
        # Encode 'season'
        for cat in season_cats:
            # The value is 1 only if the row's value matches a known category.
            encoded_row.append(1.0 if row['season'] == cat else 0.0)
            
        encoded_matrix.append(encoded_row)
    return encoded_matrix

def manual_tfidf(series, params):
    """
    Manually performs TF-IDF vectorization, matching sklearn's defaults.
    """
    vocab = params['vocabulary_']
    idf = params['idf_']
    num_vocab = len(vocab)
    
    # sklearn's default token pattern is r"(?u)\b\w\w+\b"
    # We will use a simplified version that is functionally equivalent here.
    token_pattern = re.compile(r'\b\w\w+\b')

    tfidf_matrix = []
    for text in series:
        # 1. Lowercase the text, just like sklearn does by default.
        text = text.lower()
        
        # 2. Find all tokens
        tokens = token_pattern.findall(text)
        
        # 3. Calculate term frequencies (TF)
        tf = [0.0] * num_vocab
        if len(tokens) > 0:
            word_counts = {}
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
            
            for word, count in word_counts.items():
                if word in vocab:
                    tf[vocab[word]] = count / len(tokens)
        
        # 4. Calculate TF-IDF
        tfidf_row = [tf[i] * idf[i] for i in range(num_vocab)]
        
        # 5. L2 normalization (same as before)
        norm = math.sqrt(sum(x**2 for x in tfidf_row))
        if norm > 0:
            tfidf_row = [x / norm for x in tfidf_row]
        
        tfidf_matrix.append(tfidf_row)
    return tfidf_matrix

def manual_preprocess(df, components):
    """Combines all manual preprocessing steps for ALL features."""
    
    # 1. Numerical features (processing BOTH 'impression_rating' and 'price')
    numerical_features = ['impression_rating', 'price']
    scaled_numerical = manual_impute_and_scale(df[numerical_features], components['numeric'])

    # 2. Categorical features
    categorical_features = ['room', 'season']
    ohe_matrix = manual_one_hot_encode(df[categorical_features], components['onehot'])

    # 3. Text features
    tfidf_matrix = manual_tfidf(df['combined_text'], components['tfidf'])

    # 4. Combine all features into a single matrix
    combined_matrix = []
    for i in range(len(df)):
        # The order must be exactly what the model was trained on.
        # Order: numeric, categorical, text
        row = scaled_numerical[i] + ohe_matrix[i] + tfidf_matrix[i]
        combined_matrix.append(row)
        
    return combined_matrix

# --- Feature Engineering and Prediction Logic (from before) ---

def feature_engineer(df):
    """
    Performs feature engineering. Renaming and cleaning.
    """
    rename_map = {
        'how intense is the emotion': 'impression_rating',
        'how much (in canadian dollars)': 'price',
        'which room would you put that painting in': 'room',
        'what season does this art piece remind you of': 'season',
        'how does this painting make you feel': 'moods_text',
        'what is a story that this painting tells': 'story_text'
    }
    actual_rename_dict = {}
    for col in df.columns:
        for keyword, new_name in rename_map.items():
            if keyword in col.lower():
                actual_rename_dict[col] = new_name
                break
    df = df.rename(columns=actual_rename_dict)

    def clean_price(price):
        if isinstance(price, str):
            price = re.sub(r"[^0-9.]", "", price)
            try: return float(price)
            except (ValueError, TypeError): return None 
        return price 

    df['price'] = df['price'].apply(clean_price)
    # DO NOT IMPUTE HERE. Imputation must use training set data.
    
    text_cols = ["moods_text", "story_text"]
    existing_text_cols = [col for col in text_cols if col in df.columns]
    if not existing_text_cols:
        df['combined_text'] = ''
    else:
        df['combined_text'] = df[existing_text_cols].fillna('').agg(' '.join, axis=1)
        
    return df

def predict_single_tree(tree_data, single_row):
    node_index = 0
    while tree_data['children_left'][node_index] != tree_data['children_right'][node_index]:
        feature_index = tree_data['feature'][node_index]
        threshold = tree_data['threshold'][node_index]
        if single_row[feature_index] <= threshold:
            node_index = tree_data['children_left'][node_index]
        else:
            node_index = tree_data['children_right'][node_index]
    leaf_values = tree_data['value'][node_index][0]
    return leaf_values.index(max(leaf_values))

# --- Main Prediction Function ---

def predict_all(filename):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        return f"Error: The file '{filename}' was not found."
    
    # 1. Feature Engineering
    df_engineered = feature_engineer(df)

    # 2. Manual Preprocessing
    X_processed = manual_preprocess(df_engineered, preprocessor_components)

    # 3. Manual Random Forest Prediction
    final_predictions_encoded = []
    for row in X_processed:
        tree_predictions = []
        for tree in forest_structure:
            prediction = predict_single_tree(tree, row)
            tree_predictions.append(prediction)
        majority_vote = max(set(tree_predictions), key=tree_predictions.count)
        final_predictions_encoded.append(majority_vote)

    # 4. Decode predictions using the list of classes as a lookup table
    final_predictions_decoded = [label_encoder_classes[i] for i in final_predictions_encoded]
    return final_predictions_decoded
