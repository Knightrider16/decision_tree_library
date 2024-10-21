import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Load the data
df = pd.read_csv('data.csv')

# Step 2: Preprocess the data
# Map the target variable to numeric values
df['buys_computer'] = df['buys_computer'].map({'no': 0, 'yes': 1})

# Step 3: Convert categorical variables to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 4: Define features (X) and target (y)
X = df_encoded.drop('buys_computer', axis=1)  # Features
y = df_encoded['buys_computer']                # Target

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create and train the decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['no', 'yes'])
plt.title("Decision Tree")
plt.show()


# Step 10: Validate predictions on the test set
print("\nTest Set Prediction Validation:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {'yes' if actual == 1 else 'no'}, Predicted: {'yes' if predicted == 1 else 'no'} - {'Correct' if actual == predicted else 'Incorrect'}")
