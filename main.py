from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load diabetes dataset 
X, y = load_breast_cancer(return_X_y=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y) 

# Initialize the MLP classifier
clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=1e-5, max_iter=500)

# Fit the classifier to the data
clf.fit(X_train, y_train)

# Print the accuracy on test set
print(f'Accuracy: {clf.score(X_test, y_test)}') 
