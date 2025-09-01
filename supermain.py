import streamlit as st
import webapp_v3
import webapp_v4
import extract_file_name
import roc_v1
import Logisitic_regression
import subtype
import line
import group_v1
import line


# Create a menu to select the app
app_choice = st.sidebar.selectbox("Select App", ["Chemotaxis_assay_index","Chemotaxis_index_line_graph","PCA","Chemotaxis_index_strain","Subtype", "Total_index", "ROC","Logisitic_regression","Line"])

# Run the selected app
if app_choice == "Chemotaxis_assay_index":
    webapp_v3.run(key="Chemotaxis_assay_index")  # Provide a unique key here
elif app_choice == "Chemotaxis_index_line_graph":
    line.run(key="Chemotaxis_index_line_graph")  # Provide a unique key here
elif app_choice == "PCA":
    group_v1.run(key="PCA")  # Provide a unique key here
elif app_choice == "Chemotaxis_index_strain":
    webapp_v4.run(key="Chemotaxis_index_strain")  # Provide a unique key here
elif app_choice == "Subtype":
    subtype.run(key="subtype")  # Provide a unique key here
elif app_choice == "Total_index":
    extract_file_name.run(key="EXTRACT_FILE_APP")  # Provide a unique key here
elif app_choice == "ROC":
    roc_v1.run(key="ROC_APP")  # Provide a unique key here
elif app_choice == "Logisitic_regression":
    Logisitic_regression.run(key="logreg_app")  # Provide a unique key here
elif app_choice == "Line":
    line.run(key="Line")  # Provide a unique key here
