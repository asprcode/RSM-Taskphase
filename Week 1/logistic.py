import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score

file_path = r"D:\RSM\titanic.csv"
titanic_data = pd.read_csv(file_path)

print(titanic_data.head())

titanic_data = titanic_data.drop(columns=['Name', 'Ticket', 'Cabin'])

titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'], drop_first=True)

X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = titanic_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


#tried finding other stuff, but dont fully understand how the how the code works yet for some of the following
# # Scatterplot for Predictions
# plt.figure(figsize=(8, 6))  # Ensure a new figure is created
# plt.scatter(range(len(y_test)), y_test, label="Actual", alpha=0.7, color='blue', marker='o')
# plt.scatter(range(len(y_test)), y_pred, label="Predicted", alpha=0.5, color='red', marker='x')
# plt.title("Actual vs. Predicted Labels")
# plt.xlabel("Sample Index")
# plt.ylabel("Survived (1) / Not Survived (0)")
# plt.legend()
# plt.show()  # Explicitly display the scatterplot


# # Combine actual and predicted into a single DataFrame
# results_df = pd.DataFrame({
#     'Actual': y_test.values,
#     'Predicted': y_pred
# })

# # Add a column to indicate misclassifications
# results_df['Correct'] = results_df['Actual'] == results_df['Predicted']

# # Bar plot of correct vs incorrect classifications
# plt.figure(figsize=(8, 6))
# sns.countplot(data=results_df, x='Correct', palette=['red', 'green'])
# plt.title('Correct vs. Incorrect Classifications')
# plt.xlabel('Classification')
# plt.ylabel('Count')
# plt.xticks([0, 1], labels=['Incorrect', 'Correct'])
# plt.show()

# # Optional: Categorized Scatterplot for better clarity
# plt.figure(figsize=(8, 6))
# sns.scatterplot(
#     x=range(len(results_df)), 
#     y=results_df['Actual'], 
#     hue=results_df['Correct'], 
#     palette=['red', 'green'], 
#     alpha=0.6, 
#     s=100
# )
# plt.title('Actual vs. Predicted (Highlighted Misclassifications)')
# plt.xlabel('Sample Index')
# plt.ylabel('Survived (1) / Not Survived (0)')
# plt.legend(title='Correct Prediction', labels=['Incorrect', 'Correct'])
# plt.show()


# # Select two features for visualization
# X_train_2D = X_train[['Age', 'Fare']]
# X_test_2D = X_test[['Age', 'Fare']]

# # Train a logistic regression model with these two features
# from sklearn.linear_model import LogisticRegression
# model_2D = LogisticRegression()
# model_2D.fit(X_train_2D, y_train)

# # Create a mesh grid for plotting decision boundary
# x_min, x_max = X_train_2D['Age'].min() - 1, X_train_2D['Age'].max() + 1
# y_min, y_max = X_train_2D['Fare'].min() - 1, X_train_2D['Fare'].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# # # Predict on the grid
# # grid_points = np.c_[xx.ravel(), yy.ravel()]
# # Z = model_2D.predict(grid_points).reshape(xx.shape)

# # Plot the decision boundary
# plt.figure(figsize=(10, 6))
# plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
# plt.scatter(X_test_2D['Age'], X_test_2D['Fare'], c=y_test, edgecolor='k', cmap='coolwarm', s=50)
# plt.title('Decision Boundary (Logistic Regression)')
# plt.xlabel('Age')
# plt.ylabel('Fare')
# plt.colorbar(label='Survived (1) / Not Survived (0)')
# plt.show()
