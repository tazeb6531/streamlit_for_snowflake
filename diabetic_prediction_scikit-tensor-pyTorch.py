import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.inspection import permutation_importance

import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

# Data visualization
plt.figure(figsize=(5, 3))
plt.title("Plot of Distribution of Data Per Class/Label")
df['class'].value_counts().plot(kind='bar')
plt.show()

labels = ["Less than 10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80 and more"]
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
freq_df = df.groupby(pd.cut(df['age'], bins=bins, labels=labels)).size()

plt.figure(figsize=(8, 4))
plt.bar(freq_df.index, freq_df.values)
plt.ylabel('Counts')
plt.title('Frequency Count of Age')
plt.show()

sns.boxplot(df['age'])

corr_matrix = df.corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True)
plt.show()

s = corr_matrix.abs().unstack()
top_features_per_correlation = s.sort_values(kind="quicksort")

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

# SKLEARN Section
print("\n\n----Scikit-Learn Section----")
# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
y_pred = lr_model.predict(x_test)
print("Accuracy of LR Model:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative(0)", "Positive(1)"]))
cm = ConfusionMatrixDisplay.from_estimator(lr_model, x_test, y_test)
cm.plot()
fpr, tpr, thresholds = roc_curve(y_test, lr_model.predict_proba(x_test)[:, 1])
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
precision, recall, thresholds = precision_recall_curve(y_test, lr_model.predict_proba(x_test)[:, 1])
plt.plot(recall, precision, 'b')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision-Recall Curve')
plt.show()

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
y_pred2 = dt_model.predict(x_test)
print("Accuracy of Decision Tree Model:", accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2, target_names=["Negative(0)", "Positive(1)"]))
cm = ConfusionMatrixDisplay.from_estimator(dt_model, x_test, y_test)
cm.plot()
fpr, tpr, thresholds = roc_curve(y_test, dt_model.predict_proba(x_test)[:, 1])
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
precision, recall, thresholds = precision_recall_curve(y_test, dt_model.predict_proba(x_test)[:, 1])
plt.plot(recall, precision, 'b')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision-Recall Curve')
plt.show()

# Save and Load Models
joblib.dump(lr_model, "logistic_regression_model_diabetes.pkl")
joblib.dump(dt_model, "decision_tree_model_diabetes.pkl")
loaded_lr_model = joblib.load("logistic_regression_model_diabetes.pkl")
loaded_dt_model = joblib.load("decision_tree_model_diabetes.pkl")

# Feature Importance
et_clf = ExtraTreesClassifier()
et_clf.fit(X, y)
feature_importance_df = pd.Series(et_clf.feature_importances_, index=df.columns[:-1])
feature_importance_df.nlargest(12).plot(kind='barh')
plt.show()

perm = permutation_importance(loaded_lr_model, x_test, y_test)
sorted_idx = perm.importances_mean.argsort()
plt.barh(df.columns[:-1], perm.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()

#################################
# TENSORFLOW Section
#################################
print("\n\n----TensorFlow Section----")
model = Sequential([
    Dense(16, input_dim=x_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(8, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split = 0.2, callbacks=[early_stopping])
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy}')
y_pred_prob = model.predict(x_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=["Negative(0)", "Positive(1)"]))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
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
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
plt.plot(recall, precision, 'b')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision-Recall Curve')
plt.show()

# PYTORCH Section
print("\n\n----PyTorch Section----")
class DiabetesModel(nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.fc1 = nn.Linear(x_train.shape[1], 16)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 8)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = DiabetesModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
x_train_tensor = torch.tensor(x_train, dtype = torch.float32)
x_test_tensor = torch.tensor(x_test, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype = torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype = torch.float32)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

epochs = 100
for epoch in range(epochs):
    model.train()
    for data in train_loader:
        inputs, labels = data
        labels = labels.view(-1, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    y_pred_prob = model(x_test_tensor).flatten().numpy()
    y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')
print(classification_report(y_test, y_pred, target_names=["Negative(0)", "Positive(1)"]))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
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
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
plt.plot(recall, precision, 'b')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision-Recall Curve')
plt.show()
