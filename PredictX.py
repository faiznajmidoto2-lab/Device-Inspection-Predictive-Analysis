import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Streamlit app configuration
st.set_page_config(page_title="Prediction Analysis Of Device Inspection Outcome", layout="wide")

# Title and Introduction
st.title("Prediction Analysis Of Device Inspection Outcome")

# Navigation Pane
menu = ["Upload Data", "Visualization", "Modeling and Prediction"]
choice = st.sidebar.selectbox("Select a Section", menu)

# File Upload
if choice == "Upload Data":
    uploaded_file = st.file_uploader("Upload your cleaned dataset (CSV file):", type="csv")

    if uploaded_file:
        # Load Data
        data = pd.read_csv(uploaded_file)

        # Store data in session state for later use
        st.session_state['data'] = data

        # Check for missing values
        st.write("### Missing Values Check")
        st.write(data.isnull().sum())

        # Check for duplicates
        st.subheader("Duplicate Rows")
        duplicate_count = data.duplicated().sum()
        st.write(f"Number of duplicate rows: {duplicate_count}")

        # Outlier Detection (Boxplots)
        st.subheader("Outlier Detection")
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            selected_column = st.selectbox("Select a column for boxplot (Numerical Columns Only):", numeric_cols)
            if selected_column:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=data, x=selected_column, ax=ax)
                st.pyplot(fig)
        else:
            st.write("No numeric columns available for outlier detection.")


# Visualization
if choice == "Visualization":
    st.header("Data Visualizations")
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        # Select the type of visualization
        visualization_type = st.selectbox("Select Visualization Type:", ["Bar Chart", "Pie Chart", "Correlation Heat Map"])
        
        if visualization_type == "Bar Chart":
            st.subheader("Proportion of Inspection Results")

            # Count the inspection results
            result_data = data['result'].value_counts()

            # Plot the bar chart for inspection results
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=result_data.index, y=result_data.values, palette='Paired', ax=ax)
            ax.set_title('Proportion of Inspection Results', fontsize=16)
            ax.set_xlabel('Inspection Result')
            ax.set_ylabel('Number of Inspections')
            ax.set_xticklabels(result_data.index, rotation=45)
            st.pyplot(fig)

        elif visualization_type == "Pie Chart":
            st.subheader("Top Stores by Device Count")

            if 'store_name' in data.columns:
                top_stores = data['store_name'].value_counts().head(10)

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(top_stores.values, labels=top_stores.index, autopct='%1.1f%%', startangle=140,
                       colors=sns.color_palette('Set2'))
                ax.set_title('Top 10 Stores by Device Count')
                st.pyplot(fig)
            else:
                st.write("Column 'store_name' not found for pie chart visualization.")
                
        elif visualization_type == "Correlation Heat Map":
            st.subheader("Correlation Heatmap (All Columns)")
            encoded_data = pd.get_dummies(data, drop_first=True)
            corr = encoded_data.corr()  # Compute correlation for all columns (after encoding categorical ones)

            # Plotting the heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
            ax.set_title("Correlation Heatmap (All Columns)")
            st.pyplot(fig)
                
    else:
        st.warning("Please upload and preprocess your dataset first.")

elif choice == "Modeling and Prediction":
    st.header("Modeling and Prediction")
    
    # Ensure data is available
    if 'data' in st.session_state:
        data = st.session_state['data']  # Load data from session state

        # Data Preprocessing
        st.write("### Data Preprocessing")

        # Encode categorical features
        st.write("Encoding categorical features...")
        lab = LabelEncoder()
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            data[col] = lab.fit_transform(data[col])

        # Drop non-relevant features
        data = data.drop(['deleted', 'price_verification', 'reinspection', 'state', 'device_id', 'store_id', 'inspection_id'], axis=1, errors='ignore')

        # Assign features and target
        X = data.drop(columns=['result'])
        y = data['result']

        # Handle class imbalance using SMOTE
        st.write("Applying SMOTE for class imbalance...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Data Splitting
        st.write("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Standard Scaling
        st.write("Standardizing features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model Training
        st.write("### Model Training with XGBoost")
        xgb = XGBClassifier(eval_metric='logloss', random_state=42)
        xgb.fit(X_train, y_train)

        # Model Prediction
        predictions = xgb.predict(X_test)

        # Visualization
        st.write("### Prediction Visualization")

        # Plot actual vs predicted values
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, predictions, alpha=0.5, label='Predictions')
        ax.plot(y_test, y_test, color='red', linestyle='--', label='Ideal Prediction')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs. Predicted Values')
        ax.legend()
        st.pyplot(fig)

        # Display prediction table
        st.write("#### Actual vs Predicted Values Table")
        result_df = pd.DataFrame({'Actual Values': y_test, 'Predicted Values': predictions})
        st.dataframe(result_df)
    else:
        st.warning("Please upload and preprocess your dataset first.")
