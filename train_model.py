import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys

def display_options(valid_behaviors, valid_targets, valid_locations, valid_social_contexts):
    """Display all valid options for each category"""
    print("\nValid options:")
    print(f"Behaviors: {', '.join(sorted(valid_behaviors))}")
    print(f"Targets: {', '.join(sorted(valid_targets))}")
    print(f"Locations: {', '.join(sorted(valid_locations))}")
    print(f"Social Contexts: {', '.join(sorted(valid_social_contexts))}")

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
    valid_behaviors = sorted(data["Behavior"].unique())
    valid_targets = sorted(data["Target"].unique())
    valid_locations = sorted(data["Location"].unique())
    valid_social_contexts = sorted(data["SocialContext"].unique())

    # Create mapping for outcomes
    outcome_mapping = {1: "positive", 0: "neutral", -1: "negative"}

    # Display initial options
    display_options(valid_behaviors, valid_targets, valid_locations, valid_social_contexts)

    # User input loop
    while True:
        print("\n" + "-"*50)
        print("Enter primate Behavior details (or 'quit' to exit, 'help' for options, 'data' to see combinations):")
        
        # Get behavior input
        behavior = input("Behavior (e.g., groom, fight, forage): ").strip().lower()
        if behavior == 'quit':
            break
        elif behavior == 'help':
            display_options(valid_behaviors, valid_targets, valid_locations, valid_social_contexts)
            continue
        elif behavior == 'data':
            print("\nExisting data combinations:")
            for i, row in data.iterrows():
                print(f"{i+1}. {row['Behavior']}-{row['Target']}-{row['Location']}-{row['SocialContext']} -> {outcome_mapping.get(row['Outcome'], row['Outcome'])}")
            continue
            
        # Check if behavior is valid
        if behavior not in valid_behaviors:
            print(f"Invalid behavior. Choose from: {', '.join(valid_behaviors)}")
            continue
            
        # Get target input
        target = input("Target (e.g., food, mate, rival): ").strip().lower()
        if target == 'quit':
            break
        elif target == 'help':
            display_options(valid_behaviors, valid_targets, valid_locations, valid_social_contexts)
            continue
            
        # Check if target is valid
        if target not in valid_targets:
            print(f"Invalid target. Choose from: {', '.join(valid_targets)}")
            continue
            
        # Get location input
        location = input("Location (e.g., ground, tree, nest): ").strip().lower()
        if location == 'quit':
            break
        elif location == 'help':
            display_options(valid_behaviors, valid_targets, valid_locations, valid_social_contexts)
            continue
            
        # Check if location is valid
        if location not in valid_locations:
            print(f"Invalid location. Choose from: {', '.join(valid_locations)}")
            continue

        # Get social context input
        social_context = input("Social Context (e.g., group, pair, alone): ").strip().lower()
        if social_context == 'quit':
            break
        elif social_context == 'help':
            display_options(valid_behaviors, valid_targets, valid_locations, valid_social_contexts)
            continue
            
        # Check if social context is valid
        if social_context not in valid_social_contexts:
            print(f"Invalid social context. Choose from: {', '.join(valid_social_contexts)}")
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