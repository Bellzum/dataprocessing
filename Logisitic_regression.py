import streamlit as st
import pandas as pd
import openpyxl
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go  # Import Plotly's graph objects
##########################################
def run(key):
    st.title("Regression Analysis with Streamlit")
    data_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx'], key=key)
#########################################
    
    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
                """
                Adds a UI on top of a dataframe to let viewers filter columns
                Args:
                    df (pd.DataFrame): Original dataframe
                Returns:
                    pd.DataFrame: Filtered dataframe
                """
                modify = st.checkbox("Add filters")

                if not modify:
                    return df

                df = df.copy()

                # Add a filter UI for each column in the DataFrame
                for col_name in df.columns:
                    if st.checkbox(f"Filter by {col_name}"):
                        filter_value = st.text_input(f"Enter {col_name} to filter by")
                        df = df[df[col_name] == filter_value]

                return df
    ##############################################
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

        # Display the DataFrame
        st.write("Original DataFrame:")
        st.write(df)

        # Filter the DataFrame
        filtered_df = filter_dataframe(df)

        # Check if regression or logistic regression should be performed
        analysis_type = st.selectbox("Select analysis type", ["Linear Regression", "Logistic Regression"])

        if analysis_type == "Linear Regression":
            st.subheader("Linear Regression Analysis")
            X = filtered_df[['specimen', 'conc']]
            y = filtered_df['s_index']
            X = pd.get_dummies(X, columns=['specimen', 'conc'], drop_first=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f'Mean Absolute Error: {mae}')
            st.write(f'Mean Squared Error: {mse}')
            st.write(f'R-squared: {r2}')

            # Create scatter plot for actual vs. predicted values
            st.write("Actual vs. Predicted Values")
            scatter_fig = go.Figure()  # Create a Plotly figure
            scatter_fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Scatter Plot'))
            scatter_fig.update_layout(
                xaxis_title='Actual s_index',
                yaxis_title='Predicted s_index',
                title='Actual vs. Predicted Values'
            )
            st.plotly_chart(scatter_fig)  # Display Plotly figure

             # Create histogram of prediction errors (residuals) using Plotly Express
            residuals = y_test - y_pred
            histogram_fig = px.histogram(x=residuals, title='Prediction Error Distribution')
            histogram_fig.update_xaxes(title='Residuals (Actual - Predicted)')
            histogram_fig.update_yaxes(title='Frequency')
            st.plotly_chart(histogram_fig)

        elif analysis_type == "Logistic Regression":
            st.subheader("Logistic Regression Analysis")
            X = filtered_df[['specimen', 'conc']]
            y = filtered_df['s_index']
            X = pd.get_dummies(X, columns=['specimen', 'conc'], drop_first=True)
            y = (y > y.mean()).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            predicted_labels = model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted_labels)
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            st.write(f'Accuracy: {accuracy}')
            st.write(f'ROC AUC: {roc_auc}')

            # Create ROC curve plot
            st.write("ROC Curve")
            roc_fig = px.line(x=fpr, y=tpr, title='ROC Curve')
            roc_fig.update_xaxes(title='False Positive Rate')
            roc_fig.update_yaxes(title='True Positive Rate')
            st.plotly_chart(roc_fig)  # Display Plotly figure