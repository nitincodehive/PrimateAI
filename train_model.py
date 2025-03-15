import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load primate data
data = pd.read_csv("primate_data.csv")

# Convert words to numbers
X = pd.get_dummies(data[["behavior", "context1", "context2"]])  # Features: behavior + context1 + context2
y = data["outcome"]  # What we predict: 1/0/-1

# Train model
model = LogisticRegression()
model.fit(X, y)

# Confirm the model has been trained
print("Model trained!")

# Test a modern scenario: "liking a friend’s post"
test = pd.DataFrame({"behavior": ["fight"], "context1": ["dominant_male"], "context2": ["troop"]})
test_X = pd.get_dummies(test).reindex(columns=X.columns, fill_value=0)
prediction = model.predict(test_X)
print("Prediction for 'fight-dominant_male-troop':", prediction[0])