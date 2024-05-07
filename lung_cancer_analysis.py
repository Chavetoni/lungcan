import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import tensorflow as keras
from keras.models import Sequential
from keras.layers import Dense, Input

import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\Users\dgarc\Desktop\personal\lungcan\lung_cancer.csv') 

# Check for missing values
missing_values = data.isnull().sum()
print('Missing Values:')
print(missing_values)

# Convert 'GENDER' to numerical format: M=0, F=1
data['GENDER'] = data['GENDER'].map({'M': 0, 'F': 1})

# Convert 'LUNG_CANCER' to numerical format: YES=1, NO=0
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Splitting the dataset into features (X) and target (y)
X = data.drop('LUNG_CANCER', axis=1)

y = data['LUNG_CANCER']

# Split the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Neural Network
# scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
model = Sequential([
    Input(shape=(15,)),
    Dense(3, activation='relu'),
    Dense(3, activation='relu'),
    Dense(3, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs= 100, validation_split=0.2)

# Apply SMOTE to the training data
SMOTE = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = SMOTE.fit_resample(X_train, y_train)

# Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train_resampled, y_train_resampled)

# Predict the labels for the test set
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Detailed performance analysis
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# evalutate the model
loss, accuracy = model.evaluate(X_test_scaled,y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')



##########################

# Predict the probabilities for the test set
dt_y_pred_prob = clf.predict_proba(X_test)[:, 1]
nn_y_pred_prob = model.predict(X_test_scaled).ravel()

# Calculate the ROC curves and AUC scores
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_y_pred_prob)
dt_auc = auc(dt_fpr, dt_tpr)

nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_y_pred_prob)
nn_auc = auc(nn_fpr, nn_tpr)

# Plot the ROC curves
plt.figure(figsize=(8, 6))
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_auc:.4f})')
plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing random classifier
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()