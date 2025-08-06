# 🚢 Titanic Survival Prediction using Logistic Regression

This project builds a **Logistic Regression model** to predict whether a passenger survived the Titanic disaster based on features such as age, gender, ticket class, and more.

## 📁 Dataset

The dataset used is `Titanic.csv` and includes the following features:

- **Survived**: Target variable (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Sex**: Gender (encoded: male = 0, female = 1)
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Passenger fare
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton – encoded numerically)

## ⚙️ Workflow

1. **Data Cleaning**:
   - Dropped irrelevant columns: `Ticket`, `Cabin`, `Name`
   - Removed rows with missing `Age` or `Embarked` values

2. **Feature Engineering**:
   - Encoded categorical features: `Sex`, `Embarked`
   - Selected useful features for training

3. **Model Training**:
   - Split the dataset into training and testing sets (80/20)
   - Trained a logistic regression model using `sklearn`

4. **Evaluation**:
   - Evaluated using Accuracy Score, Confusion Matrix, and Classification Report (precision, recall, F1-score)

## 📊 Results

The model prints:
- **Accuracy Score**: Overall performance
- **Confusion Matrix**: Actual vs Predicted values
- **Classification Report**: Includes precision, recall, F1-score for each class

## 🧠 Technologies Used

- Python 3.x
- pandas
- scikit-learn

Install dependencies (if needed):
```bash
pip install pandas scikit-learn
