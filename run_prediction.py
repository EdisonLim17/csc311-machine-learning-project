#local script to test pred.py

import pandas as pd
from pred import predict_all


test_filename = 'test_data.csv'


predictions = predict_all(test_filename)


if isinstance(predictions, list):
    try:
        # Load the test data to get the true labels
        test_df = pd.read_csv(test_filename)
        true_labels = test_df['artist'].tolist()

        # Compare predictions to true labels
        correct_predictions = 0
        total_predictions = len(predictions)

        if total_predictions > 0:
            for pred, true in zip(predictions, true_labels):
                if pred == true:
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_predictions
            
            print(f"Successfully generated {total_predictions} predictions.")
            print(f"Accuracy on test set: {accuracy:.4f} ({correct_predictions}/{total_predictions} correct)")
            print("\nFirst 5 predictions:", predictions[:5])
            print("First 5 true labels:", true_labels[:5])

        else:
            print("Generated an empty list of predictions.")

    except (FileNotFoundError, KeyError) as e:
        print(f"Error calculating accuracy: {e}")
        print("Please ensure 'test_data.csv' exists and contains an 'artist' column.")

else:
    print(predictions)
