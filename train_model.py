import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys

def main():
    # Load primate data
    try:
        data = pd.read_csv("primate_data.csv")
    except FileNotFoundError:
        print("Error: primate_data.csv file not found")
        sys.exit(1)

    # Convert words to numbers
    X = pd.get_dummies(data[["Behavior", "Target", "Location", "SocialContext"]])  # Features
    y = data["Outcome"]  # Target

    # Train model
    model = LogisticRegression()
    model.fit(X, y)
    print("Model trained!")

    # Store valid values for each feature
    valid_behaviors = data["Behavior"].unique()
    valid_targets = data["Target"].unique()
    valid_locations = data["Location"].unique()
    valid_social_contexts = data["SocialContext"].unique()

    # Create mapping for outcomes
    outcome_mapping = {1: "positive", 0: "neutral", -1: "negative"}

    # User input loop
    while True:
        print("\n" + "-"*50)
        print("Enter primate Behavior details (or 'quit' to exit):")
        
        # Get behavior input
        behavior = input("Behavior (e.g., groom, fight, forage): ").strip().lower()
        if behavior == 'quit':
            break
            
        # Check if behavior is valid
        if behavior not in valid_behaviors:
            print("I cannot compute. Need more data.")
            continue
            
        # Get target input
        target = input("Target (e.g., friend, rival, dominant_male): ").strip().lower()
        if target == 'quit':
            break
            
        # Check if target is valid
        if target not in valid_targets:
            print("I cannot compute. Need more data.")
            continue
            
        # Get location input
        location = input("Location (e.g., troop, food): ").strip().lower()
        if location == 'quit':
            break
            
        # Check if location is valid
        if location not in valid_locations:
            print("I cannot compute. Need more data.")
            continue

        # Get social context input
        social_context = input("Social Context (e.g., group, solitary): ").strip().lower()
        if social_context == 'quit':
            break
            
        # Check if social context is valid
        if social_context not in valid_social_contexts:
            print("I cannot compute. Need more data.")
            continue
        
        # Create test data
        test = pd.DataFrame({
            "Behavior": [behavior],
            "Target": [target],
            "Location": [location],
            "SocialContext": [social_context]
        })
        
        # Transform test data to match training format
        test_X = pd.get_dummies(test).reindex(columns=X.columns, fill_value=0)
        
        # Check if any features are missing (all zeros would indicate unknown combination)
        if test_X.sum().sum() == 0:
            print("I cannot compute. Need more data.")
            continue
            
        # Make prediction
        prediction = model.predict(test_X)[0]
        
        # Convert numeric prediction to text
        result = outcome_mapping[prediction]
        
        # Display result
        print(f"\nPrediction for '{behavior}-{target}-{location}-{social_context}': {result}")

if __name__ == "__main__":
    main()