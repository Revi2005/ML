# K-Nearest Neighbours (K-NN) Algorithm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create KNN Classifier
k = 3
model = KNeighborsClassifier(n_neighbors=k)

# Train the model
model.fit(X_train, y_train)

# Predict the test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("K-Nearest Neighbours (KNN)")
print("-----------------------------------")
print("Value of K =", k)

print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy = {:.2f}%".format(accuracy * 100))

# Predict a new sample
new_sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_sample)

print("\nNew Sample:", new_sample)

print("Predicted Class:", iris.target_names[prediction[0]])