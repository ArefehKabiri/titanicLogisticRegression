# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the Titanic dataset from a CSV file
df = pd.read_csv('Titanic.csv')

# Drop columns that are not useful for prediction
df = df.drop(['Ticket', 'Cabin', 'Name'], axis=1)

# Remove rows where 'Age' or 'Embarked' is missing
df = df.dropna(subset=['Age', 'Embarked'])

# Convert 'Sex' column to numerical values: male = 0, female = 1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Encode 'Embarked' column as numerical values
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# Define input features (X) and target variable (y)
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict the survival on the test set
y_pred = model.predict(X_test)

# Print the accuracy score of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print the confusion matrix to see true vs predicted values
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Print a detailed classification report (precision, recall, F1-score)
print("Classification Report:\n", classification_report(y_test, y_pred))
