import pandas as pd
import joblib
import re

# Load the trained model and preprocessing objects from the files
try:
    model = joblib.load('final_model.joblib')
    preprocessor = joblib.load('final_preprocessor.joblib')
    label_encoder = joblib.load('final_label_encoder.joblib')
except FileNotFoundError:
    print("Error: Make sure 'final_model.joblib', 'final_preprocessor.joblib', and 'final_label_encoder.joblib' are in the same directory.")
    exit()

def feature_engineer(df):
    """
    Performs the same feature engineering as in the training notebook.
    """
    # 1. Rename columns for simplicity
    df = df.rename(columns={
        "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": "impression_rating",
        "How much (in Canadian dollars) would you be willing to pay for this painting?": "price",
        "If you could purchase this painting, which room would you put that painting in?": "room",
        "What season does this art piece remind you of?": "season"
    })

    # 2. Clean the 'price' column
    def clean_price(price):
        if isinstance(price, str):
            # Remove non-numeric characters except for the decimal point
            price = re.sub(r"[^0-9.]", "", price)
            try:
                return float(price)
            except (ValueError, TypeError):
                return None # Return None if conversion fails
        return price # Return as is if it's already a number

    df['price'] = df['price'].apply(clean_price)

    # 3. Combine text columns into a single feature
    text_cols = [
        "Describe how this painting makes you feel.",
        "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."
    ]
    # Fill any missing text with an empty string before combining
    df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)

    return df


def predict_all(filename):
    """
    Reads a CSV file, engineers features, preprocesses the data,
    and returns predictions for each row.
    """
    # Read the file containing the test data
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        return f"Error: The file '{filename}' was not found."

    # Ensure the dataframe is not empty
    if df.empty:
        return []

    # **Perform the crucial feature engineering step**
    df_engineered = feature_engineer(df)

    # Preprocess the engineered data using the loaded preprocessor
    X_processed = preprocessor.transform(df_engineered)

    # Make predictions on the processed data
    predictions_encoded = model.predict(X_processed)

    # Convert the numerical predictions back to their original string labels
    predictions_decoded = label_encoder.inverse_transform(predictions_encoded)

    return predictions_decoded.tolist()

# Example of how to run the script (optional, for your own testing)
if __name__ == '__main__':
    test_filename = 'test_sample.csv'
    predictions = predict_all(test_filename)
    if isinstance(predictions, list):
        print(f"Predictions for {test_filename}:")
        print(predictions)
