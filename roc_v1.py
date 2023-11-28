import pandas as pd
import streamlit as st
import openpyxl
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from io import BytesIO
from typing import List
import altair as alt
import base64
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import streamlit.components.v1 as html
from PIL import Image
from lib2to3.pgen2.pgen import DFAState
############################################    
def run(key):
    st.title("ROC")
    data_file = st.sidebar.file_uploader("Upload Excel file for ROC", type=['xlsx'], key=key)
    
    if data_file:
            
            
            file_details = {
                "Filename": data_file.name,
                "FileType": data_file.type,
                "FileSize": data_file.size
            }

            wb = openpyxl.load_workbook(data_file)

            # Show file details and sheet selector in sidebar
            st.sidebar.subheader("File details:")
            st.sidebar.json(file_details, expanded=False)
            st.sidebar.markdown("----")
            sheet_selector = st.sidebar.selectbox("Select sheet:", wb.sheetnames)

            # Read selected sheet into a Pandas DataFrame
            df = pd.read_excel(data_file, sheet_name=sheet_selector)
            st.markdown(f"### Currently Selected worksheet: `{sheet_selector}`")

            concentration_column = 'conc'
            unique_concentrations = df[concentration_column].unique()

            # Initialize models
            model_lr = LogisticRegression(C=1, random_state=42, solver='lbfgs')
            model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

            # Collect probabilities and true labels for all concentrations
            proba_lr_all = []
            proba_rf_all = []
            y_test_all = []

            for concentration in unique_concentrations:
                df_concentration = df[df[concentration_column] == concentration]
                    
                X_concentration = df_concentration['s_index'].values
                y_concentration = df_concentration['cancer'].values

                X_train, X_test, y_train, y_test = train_test_split(X_concentration, y_concentration, test_size=0.3, random_state=1, stratify=y_concentration)

                model_lr.fit(X_train.reshape(-1, 1), y_train)
                model_rf.fit(X_train.reshape(-1, 1), y_train)

                proba_lr = model_lr.predict_proba(X_test.reshape(-1, 1))[:, 1]
                proba_rf = model_rf.predict_proba(X_test.reshape(-1, 1))[:, 1]

                proba_lr_all.extend(proba_lr)
                proba_rf_all.extend(proba_rf)
                y_test_all.extend(y_test)

                # Calculate overall AUC scores
                auc_lr_all = roc_auc_score(y_test_all, proba_lr_all)
                auc_rf_all = roc_auc_score(y_test_all, proba_rf_all)

                # Calculate overall ROC curve
                fpr_lr_all, tpr_lr_all, _ = roc_curve(y_test_all, proba_lr_all)
                fpr_rf_all, tpr_rf_all, _ = roc_curve(y_test_all, proba_rf_all)

                interp_fpr_all = np.linspace(0, 1, 100)
                interp_tpr_lr_all = np.interp(interp_fpr_all, fpr_lr_all, tpr_lr_all)
                interp_tpr_rf_all = np.interp(interp_fpr_all, fpr_rf_all, tpr_rf_all)

                roc_data_overall = pd.DataFrame({
                    'FPR_LR': interp_fpr_all,
                    'TPR_LR': interp_tpr_lr_all,
                    'FPR_RF': interp_fpr_all,
                    'TPR_RF': interp_tpr_rf_all
                })

                fig_overall = px.line(roc_data_overall, x='FPR_LR', y='TPR_LR', title='Overall ROC Curve')
                fig_overall.add_scatter(x=roc_data_overall['FPR_RF'], y=roc_data_overall['TPR_RF'], mode='lines', name='Random Forest')
                fig_overall.update_layout(
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                )

                st.write("### Overall ROC Curve")
                st.plotly_chart(fig_overall)
                
                # Display overall AUC scores
                st.write("### Overall AUC Scores")
                st.write(f'Logistic Regression AUC (Overall): {auc_lr_all:.4f}')
                st.write(f'Random Forest AUC (Overall): {auc_rf_all:.4f}')

            # Create a list to store AUC data
            auc_data = []

            # Loop over each concentration
            for concentration in unique_concentrations:
                st.write(f"### Concentration: {concentration}")

                df_concentration = df[df[concentration_column] == concentration]
                X = df_concentration['s_index'].values
                y = df_concentration['cancer'].values

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

                model_lr = LogisticRegression(C=1, random_state=42, solver='lbfgs')
                model_lr.fit(X_train.reshape(-1, 1), y_train)

                model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
                model_rf.fit(X_train.reshape(-1, 1), y_train)

                proba_lr = model_lr.predict_proba(X_test.reshape(-1, 1))[:, 1]
                proba_rf = model_rf.predict_proba(X_test.reshape(-1, 1))[:, 1]

                fpr_lr, tpr_lr, _ = roc_curve(y_test, proba_lr)
                fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)

                interp_fpr = np.linspace(0, 1, 100)
                interp_tpr_lr = np.interp(interp_fpr, fpr_lr, tpr_lr)
                interp_tpr_rf = np.interp(interp_fpr, fpr_rf, tpr_rf)

                roc_data = pd.DataFrame({'FPR_LR': interp_fpr, 'TPR_LR': interp_tpr_lr, 'FPR_RF': interp_fpr, 'TPR_RF': interp_tpr_rf})

                fig = px.line(roc_data, x='FPR_LR', y='TPR_LR', title=f'ROC Curve for Concentration: {concentration}')
                fig.add_scatter(x=roc_data['FPR_RF'], y=roc_data['TPR_RF'], mode='lines', name='Random Forest')
                fig.update_layout(
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                )
                st.plotly_chart(fig)

                st.write(f'Logistic Regression AUC: {roc_auc_score(y_test, proba_lr):.4f}')
                st.write(f'Random Forest AUC: {roc_auc_score(y_test, proba_rf):.4f}')

                # Calculate AUC scores for the current concentration
                auc_lr = roc_auc_score(y_test, proba_lr)
                auc_rf = roc_auc_score(y_test, proba_rf)

                # Append AUC data to the list
                auc_data.append({
                    'Concentration': concentration,
                    'Logistic Regression AUC': auc_lr,
                    'Random Forest AUC': auc_rf
                })

            # Calculate overall AUC scores
            auc_lr_all = roc_auc_score(y_test_all, proba_lr_all)
            auc_rf_all = roc_auc_score(y_test_all, proba_rf_all)

            # Append overall AUC data to the list
            auc_data.append({
                'Concentration': 'Overall',
                'Logistic Regression AUC': auc_lr_all,
                'Random Forest AUC': auc_rf_all
            })

            # Create a DataFrame from the collected AUC data
            auc_df = pd.DataFrame(auc_data)

            # Display the AUC data table
            st.write("### AUC Data Table")
            st.dataframe(auc_df)

