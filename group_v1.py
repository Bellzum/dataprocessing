import streamlit as st
import pandas as pd
import openpyxl
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

def run(key):
    st.sidebar.title("Upload File")
    data_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx'], key=key)

    if data_file:
        try:
            # Display file details
            file_details = {
                "Filename": data_file.name,
                "FileType": data_file.type,
                "FileSize": data_file.size
            }
            st.sidebar.subheader("File details:")
            st.sidebar.json(file_details, expanded=False)
            st.sidebar.markdown("----")

            # Load workbook and select sheet
            wb = openpyxl.load_workbook(data_file)
            sheet_selector = st.sidebar.selectbox("Select sheet:", wb.sheetnames)
            df = pd.read_excel(data_file, sheet_selector)

            st.markdown(f"### Currently Selected Worksheet: `{sheet_selector}`")
            st.dataframe(df)

            # Ensure required columns exist
            required_columns = ['specimen', 'conc', 's_index']
            if not all(col in df.columns for col in required_columns):
                st.error(f"The selected sheet must contain the following columns: {required_columns}")
                return

            # Encode specimen labels
            le = LabelEncoder()
            df['specimen_encoded'] = le.fit_transform(df['specimen'])

            # Prepare data for PCA
            df['conc_numeric'] = pd.Categorical(df['conc']).codes  # Encode concentrations numerically
            features = ['s_index', 'conc_numeric']
            X = df[features]
            y = df['specimen_encoded']

            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # Create a PCA DataFrame
            pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
            pca_df['Group'] = le.inverse_transform(y)

            # Visualize PCA
            st.header("PCA Visualization")
            fig = px.scatter(
                pca_df, x='PC1', y='PC2', color='Group', title="PCA of Concentrations",
                labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Balance the dataset using SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_pca, y)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

            # Train LightGBM Classifier
            clf = LGBMClassifier(random_state=42)
            clf.fit(X_train, y_train)

            # Make predictions
            y_pred = clf.predict(X_test)

            # Display classification results
            st.header("Classification Results")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred, target_names=le.classes_))
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    run(key="chemotaxis_assay_index")

