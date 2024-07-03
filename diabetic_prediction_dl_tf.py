import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
df = pd.read_csv("first_app/data/diabetes_data_upload.csv")
df.columns = df.columns.str.lower().str.replace(' ','_')

# Label Encoding
columns_to_label_encode = ['polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness',
                           'polyphagia', 'genital_thrush', 'visual_blurring', 'itching',
                           'irritability', 'delayed_healing', 'partial_paresis',
                           'muscle_stiffness', 'alopecia', 'obesity']

LE = LabelEncoder()
for col in columns_to_label_encode:
    df[col] = LE.fit_transform(df[col].astype(str))

gender_map = {"Female": 0, "Male": 1}
target_label_map = {"Negative": 0, "Positive": 1}
df['gender'] = df['gender'].map(gender_map)
df['class'] = df['class'].map(target_label_map)

# Split the data into features and target
X = df[['age', 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss',
        'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring',
        'itching', 'irritability', 'delayed_healing', 'partial_paresis',
        'muscle_stiffness', 'alopecia', 'obesity']]
y = df['class']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)

# Define the model
model = Sequential([
    Dense(16, input_dim=x_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(8, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Predictions
y_pred_prob = model.predict(x_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

y_pred

# Classification Report
print(classification_report(y_test, y_pred, target_names=["Negative(0)", "Positive(1)"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
plt.plot(recall, precision, 'b')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision-Recall Curve')
plt.show()

