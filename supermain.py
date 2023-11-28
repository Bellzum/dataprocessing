import streamlit as st
import webapp_v3
import extract_file_name
import roc_v1
import Logisitic_regression

# Create a menu to select the app
app_choice = st.sidebar.selectbox("Select App", ["Chemotaxis_assay_index", "Total_index", "ROC","Logisitic_regression"])

# Run the selected app
if app_choice == "Chemotaxis_assay_index":
    webapp_v3.run(key="Chemotaxis_assay_index")  # Provide a unique key here
elif app_choice == "Total_index":
    extract_file_name.run(key="EXTRACT_FILE_APP")  # Provide a unique key here
elif app_choice == "ROC":
    roc_v1.run(key="ROC_APP")  # Provide a unique key here
elif app_choice == "Logisitic_regression":
    Logisitic_regression.run(key="Logisitic_regression")  # Provide a unique key here