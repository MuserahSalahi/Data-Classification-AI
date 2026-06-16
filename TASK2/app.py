import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Page configuration - Professional Layout
st.set_page_config(page_title="AI Data Classification Dashboard", layout="wide")

# ==========================================
# DATA & MODEL CORE SETUP (Caching for Speed)
# ==========================================
@st.cache_data
def load_and_train_pipeline():
    iris_data = load_iris()
    X = iris_data.data
    y = iris_data.target
    
    # Train-Test Split & Scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    return iris_data, scaler, knn, X_train_scaled, y_train, X_test_scaled, y_test

iris_data, scaler, knn_model, X_train_scaled, y_train, X_test_scaled, y_test = load_and_train_pipeline()

# ==========================================
# DASHBOARD UI SIDEBAR: User Inputs
# ==========================================
st.sidebar.header(" Input Iris Dimensions (cm)")

# Exact feature names based on dataset benchmarks
sepal_l = st.sidebar.slider("Sepal Length", float(iris_data.data[:, 0].min()), float(iris_data.data[:, 0].max()), 5.1)
sepal_w = st.sidebar.slider("Sepal Width", float(iris_data.data[:, 1].min()), float(iris_data.data[:, 1].max()), 3.5)
petal_l = st.sidebar.slider("Petal Length", float(iris_data.data[:, 2].min()), float(iris_data.data[:, 2].max()), 1.4)
petal_w = st.sidebar.slider("Petal Width", float(iris_data.data[:, 3].min()), float(iris_data.data[:, 3].max()), 0.2)

# ==========================================
# MAIN DASHBOARD CONTENT
# ==========================================
st.title("Data Classification Using AI")
st.markdown("### Supervised Learning Pipeline via K-Nearest Neighbors (KNN)")

# Layout Columns: Left for Prediction, Right for Metrics Visualization
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(" Model Inference")
    
    # User inputs ko array mein convert karke scale karna
    user_input = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    user_input_scaled = scaler.transform(user_input)
    
    # Predict target class
    prediction = knn_model.predict(user_input_scaled)
    predicted_species = iris_data.target_names[prediction[0]]
    
    # Display Result inside a clean card structure
    st.info(f"**Predicted Species:** Class {prediction[0]} - **{predicted_species.upper()}**")
    
    # Feature Preview Table
    st.markdown("#### Current Inputs Table")
    input_df = pd.DataFrame(user_input, columns=["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"])
    st.dataframe(input_df, hide_index=True)


with col2:
    st.subheader(" Model Evaluation & Logic Boundaries")
    
    # Figure out predictions and confusion matrix on test set 
    y_pred = knn_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    # Figure setup
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Generate Heatmap  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris_data.target_names, yticklabels=iris_data.target_names, ax=ax)
    
    predicted_index = prediction[0]
    
    # ----------------------------------------------------
    #  1: Row Highlight (Rectangle Border Box)
    # ----------------------------------------------------
    import matplotlib.patches as patches
    # Rectangle(x_start, y_start, width, height)
    rect = patches.Rectangle((0, predicted_index), 3, 1, linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    # ----------------------------------------------------
    #  2: Dynamic Text Labels (Red & Bold)
    # ----------------------------------------------------
    for i, label in enumerate(ax.get_yticklabels()):
        if i == predicted_index:
            label.set_color('red')       # Surkh rang
            label.set_weight('bold')     # Mota/Bold text
            label.set_fontsize(12)       # Bada size
            
    # X-axis ke labels ko clear aur neat rakhne ke liye
    ax.set_title("Confusion Matrix", fontsize=11)
    ax.set_xlabel("Predicted Species", fontsize=9)
    ax.set_ylabel("Actual Species", fontsize=9)
    
    st.pyplot(fig)

st.divider()


