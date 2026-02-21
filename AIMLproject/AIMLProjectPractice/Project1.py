# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ---------------------------
# Step 1: Explore Data
# ---------------------------
print(train.head())
print(train.info())
print(train.isnull().sum())

# Visualize survival count
sns.countplot(x='Survived', data=train)
plt.show()

# ---------------------------
# Step 2: Handle Missing Values
# ---------------------------
# Fill missing Age with median
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)

# Fill missing Embarked with mode
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing)
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

# ---------------------------
# Step 3: Encode Categorical Variables
# ---------------------------
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

train['Embarked'] = le.fit_transform(train['Embarked'])
test['Embarked'] = le.transform(test['Embarked'])

# ---------------------------
# Step 4: Feature Selection
# ---------------------------
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train[features]
y = train['Survived']

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Step 5: Train Model
# ---------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# Step 6: Evaluate Model
# ---------------------------
y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ---------------------------
# Step 7: Make Predictions on Test Set
# ---------------------------
test_preds = model.predict(test[features])
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_preds
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")
