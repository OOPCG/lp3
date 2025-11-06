#Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load Dataset
df = pd.read_csv("Churn_Modelling.csv")  # Make sure the CSV file is in your working directory
print(" Dataset Loaded Successfully!")
print(df.head())

#Select Features and Target
# Columns 3 to 12 → features; Column 13 → target (Exited)
X = df.iloc[:, 3:13]
y = df.iloc[:, 13]

#Encode Categorical Variables
# Encode Gender (Male/Female → 0/1)
le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])

# One-Hot Encode Geography (France, Spain, Germany → 2 dummy columns)
X = pd.get_dummies(X, columns=['Geography'], drop_first=True)

#Split into Train & Test sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (important for neural networks)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Build Neural Network
model = Sequential()
model.add(Dense(units=6, activation='relu', input_dim=X_train.shape[1]))  # Input + 1st Hidden Layer
model.add(Dropout(0.3))  # Prevent overfitting
model.add(Dense(units=6, activation='relu'))  # 2nd Hidden Layer
model.add(Dense(units=1, activation='sigmoid'))  # Output Layer (sigmoid → 0/1)

#Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#  Train Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Plot Training vs Validation Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Make Predictions on Test Set
y_prob = model.predict(X_test)              # Probabilities (between 0 and 1)
y_pred = (y_prob > 0.5).astype(int).reshape(-1)  # Convert to 0 or 1

# Evaluate Model
acc = accuracy_score(y_test, y_pred)
print(f"Final Accuracy: {acc*100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
