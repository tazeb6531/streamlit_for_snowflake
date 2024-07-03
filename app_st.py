import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Function to load default data
def load_default_data():
    df = pd.read_csv("data/readmission_data.csv")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df.copy()  # Return a copy to avoid mutation
# Function to load user-uploaded data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return df.copy()  # Return a copy to avoid mutation
    else:
        return load_default_data()

def home():
    st.title("Welcome to the NYC Health + Hospitals Data Analysis App")

    # Autoplay video using Streamlit video with autoplay argument
    video_file = open('data/nychh_intro.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes, format='video/mp4', start_time=0)

st.write("""
    The leading public healthcare provider in the United States, offering vital inpatient, outpatient, and home-based services to over one million people annually at various locations.

    This application offers a robust data analysis tool designed to explore, analyze, and predict patient readmission rates using advanced machine learning and deep learning techniques.

    Navigate through the app using the panel on the left to access different sections:
    - **Home**: Overview and information about the healthcare system.
    - **EDA**: Conduct Exploratory Data Analysis on the dataset.
    - **Feature Engineering**: Process and prepare data for modeling.
    - **Model Training**: Train various machine learning models.
    - **Prediction**: Generate predictions using trained models.
""")


# EDA
def eda(df):
    st.title("Exploratory Data Analysis")
    
    st.write("### Data Overview")
    st.write(df.head())

    st.write("### Summary Statistics")
    st.write(df.describe())

    st.write("### Data Distribution per Readmission")
    fig = px.bar(df['readmission'].value_counts(), title="Distribution of Data Per Readmission")
    st.plotly_chart(fig)
    
    st.write("### Frequency Count of Age")
    fig = px.histogram(df, x='age', title="Frequency Count of Age")
    st.plotly_chart(fig)

    st.write("### Gender Distribution")
    fig = px.pie(df, names='gender', title="Gender Distribution")
    st.plotly_chart(fig)

    st.write("### BMI Distribution")
    fig = px.histogram(df, x='bmi', title="BMI Distribution")
    st.plotly_chart(fig)

    st.write("### Length of Stay Distribution")
    fig = px.histogram(df, x='length_of_stay', title="Length of Stay Distribution")
    st.plotly_chart(fig)

    st.write("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm', fmt=".2f")
    st.pyplot(fig)

    st.write("### Pairplot")
    fig = sns.pairplot(df, hue='readmission')
    st.pyplot(fig)

# Feature Engineering
def feature_engineering(df):
    columns_to_label_encode = ['diabetes', 'hypertension', 'heart_disease', 'smoking']
    
    df = df.copy()  # Avoid mutating the original dataframe
    
    LE = LabelEncoder()
    for col in columns_to_label_encode:
        df[col] = LE.fit_transform(df[col].astype(str))
    
    gender_map = {"Female": 0, "Male": 1}
    df['gender'] = df['gender'].map(gender_map)

    if 'readmission' in df.columns:
        target_map = {"No": 0, "Yes": 1}
        df['readmission'] = df['readmission'].map(target_map)
    
    return df

def display_feature_engineering(df):
    st.title("Feature Engineering")
    
    st.write("### Encoding Categorical Variables")
    df = feature_engineering(df)
    
    st.write("### Feature Engineering Completed")
    st.write(df.head())

    return df

# Model Training
def train_models(df, model_choice):
    st.title("Model Training and Evaluation")

    # Ensure that all features are numeric
    df = feature_engineering(df)
    if 'readmission' not in df.columns:
        st.error("The dataset must contain the 'readmission' column.")
        return
    
    X = df.drop(columns=['readmission'])
    y = df['readmission']
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if model_choice == "Deep Learning":
        # Define the model
        model = Sequential([
            Dense(16, input_dim=X_train.shape[1], activation='relu'),
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
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        
        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        st.write(f'Test Accuracy: {accuracy}')
        
        # Save the model
        model.save("deep_learning_model.h5")
        
        # Predictions
        y_pred_prob = model.predict(X_test).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        st.write("Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=["No", "Yes"]))
        
        st.write("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(plt)
        
        st.write("ROC Curve")
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
        st.pyplot(plt)
        
        st.write("Precision-Recall Curve")
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
        plt.plot(recall, precision, 'b')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.title('Precision-Recall Curve')
        st.pyplot(plt)
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }
        model = models[model_choice]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### {model_choice}")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Save the model
        model_filename = f"{model_choice.replace(' ', '_')}.joblib"
        joblib.dump(model, model_filename)

# Prediction
def prediction(df, model_choice):
    st.title("Prediction")
    
    st.write("### Input Features")
    
    # Determine the boundaries for the age input
    min_age = int(df['age'].min())
    max_age = int(df['age'].max())
    
    # Determine the boundaries for the numerical inputs
    numerical_columns = ['age', 'bmi', 'length_of_stay', 'num_medications', 'num_procedures']
    numerical_bounds = {col: (int(df[col].min()), int(df[col].max())) for col in numerical_columns}
    
    input_data = {}
    for col in df.columns:
        if col != 'readmission':
            if col in numerical_columns:
                min_val, max_val = numerical_bounds[col]
                input_data[col] = st.slider(col, min_value=min_val, max_value=max_val, value=min_val)
            elif df[col].dtype == object or len(df[col].unique()) <= 10:  # Assuming categorical if less than 10 unique values
                unique_values = df[col].unique()
                input_data[col] = st.selectbox(col, options=unique_values)
            else:
                input_data[col] = st.number_input(col, value=0)
    
    input_df = pd.DataFrame([input_data])
    
    if st.button("Predict"):
        # Ensure the input data matches the training data format
        df = feature_engineering(df)
        input_df = feature_engineering(input_df)
        
        if 'readmission' not in df.columns:
            st.error("The dataset must contain the 'readmission' column.")
            return

        # Standardize the input features
        scaler = StandardScaler().fit(df.drop(columns=['readmission']))
        input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        
        try:
            if model_choice == "Deep Learning":
                model = tf.keras.models.load_model("deep_learning_model.h5")
                prediction_prob = model.predict(input_df).flatten()
                prediction = (prediction_prob > 0.5).astype(int)
            else:
                model_filename = f"{model_choice.replace(' ', '_')}.joblib"
                model = joblib.load(model_filename)
                prediction = model.predict(input_df)
            
            st.write(f"Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
        except FileNotFoundError as e:
            st.error(f"Model file not found: {e}. Please ensure the model is trained and saved correctly.")
        except ValueError as e:
            st.error(f"Value error: {e}. Please ensure the input values are correct and match the training data format.")

def main():
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Home", "EDA", "Feature Engineering", "Model Training", "Prediction"])
    
    # File uploader for user-uploaded data
    st.sidebar.title("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    
    # Apply custom CSS for NYCHH color scheme
    st.markdown("""
        <style>
        .reportview-container {
            background: #f0f2f6;
            color: #333;
        }
        .sidebar .sidebar-content {
            background: #005eb8;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    df = load_data(uploaded_file)
    
    if options == "Home":
        home()
    elif options == "EDA":
        eda(df)
    elif options == "Feature Engineering":
        display_feature_engineering(df)
    elif options == "Model Training":
        model_choice = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Decision Tree", "Random Forest", "Deep Learning"])
        train_models(df, model_choice)
    elif options == "Prediction":
        model_choice = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Decision Tree", "Random Forest", "Deep Learning"])
        prediction(df, model_choice)

if __name__ == "__main__":
    main()
