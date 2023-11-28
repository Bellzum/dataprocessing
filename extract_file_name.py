from io import BytesIO
from typing import List

import altair as alt
import base64
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm
import streamlit as st
import streamlit.components.v1 as html
from PIL import Image
from lib2to3.pgen2.pgen import DFAState

#############################################
def run(key):
    st.sidebar.title("Upload File")
    data_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx'], key=key)
    
############################################    

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
        # Add a download button to export the current dataframe to an Excel file
    def download_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1')
        writer.save()
        processed_data = output.getvalue()
        return processed_data
    ################################################
    # Define a dictionary of cancer names and their corresponding keywords
    cancer_dict = {'Bileduct':'Bileduct','Breast': 'Breast','Cervical': 'Cervical',
                'Colon (大腸)': 'Colon(大腸)','Colon(結腸)': 'Colon(結腸)',
                'Endometrium': 'Endometrium','Esophageal': 'Esophageal', 
                'Gastric': 'Gastric','Glottis':'Glottis',
                'Liver': 'Liver', 'Lung': 'Lung',
                'Ovarian': 'Ovarian','Pancreatic': 'Pancreatic', 
                'Pharyngeal':'Pharyngeal',
                'Prostate': 'Prostate','Testicular':'Testicular',
                'Thyroid': 'Thyroid','Tongue':'Tongue',
                'Urinary': 'Urinary','Healthy': 'Healthy'
                    }       
    # Define a dictionary of stages and their corresponding keywords
    stage_dict = {'stage_I': r'\b1\b', 'stage_II': r'\b2\b', 'stage_III': r'\b3\b', 'stage_IV': r'\b4\b', 'Healthy':r'\b5\b'}

    st.sidebar.write("[🔗 Batch Dataset analyzer](http://192.168.6.65:8501)", unsafe_allow_html=True, style={'font-size': '30px'})


    if data_file:
        file_details = {
            "Filename":data_file.name,
            "FileType":data_file.type,
            "FileSize":data_file.size
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
        



        # Apply filters to the DataFrame
        modify = st.checkbox("Add filters")
        if modify:
            df = df.loc[df.astype(str).drop_duplicates().index]

            # Add a filter UI for each column in the DataFrame
            for col_name in df.columns:
                if st.checkbox(f"Filter by {col_name}"):
                    filter_value = st.text_input(f"Enter {col_name} to filter by")
                    df = df[df[col_name] == filter_value]


        # Loop through each row in the DataFrame and insert new rows based on cancer keywords
        
        df['cancer_name'] = ''
        df['stage'] = ''
        for index, row in df.iterrows():
            for cancer_name, keyword in cancer_dict.items():
                if keyword in row['specimen']:
                    df.at[index, 'cancer_name'] = cancer_name
                    # Extract stage from specimen column
                    specimen_stage = row['specimen'].split('_')[-1]
                    # Set the stage in the new column
                    df.at[index, 'stage'] = specimen_stage
                    break

        # Reorder the columns of the DataFrame
        df = df[['s_index'] + [col for col in df.columns if col != 's_index']]
        df = df[['conc'] + [col for col in df.columns if col != 'conc']]
        df = df[['stage'] + [col for col in df.columns if col != 'stage']]
        df = df[['cancer_name'] + [col for col in df.columns if col != 'cancer_name']]
        


        if st.button('Download Excel'):
            processed_data = download_excel(df)
            b64 = base64.b64encode(processed_data)
            href = f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="output.xlsx">Download Excel file</a>'
            st.markdown(href, unsafe_allow_html=True)
        

    
        # Display the updated DataFrame
        st.write(df)
        df_mean= df.groupby(['cancer_name']).agg({"s_index":["mean"]})
        df_cal=df.groupby(['cancer_name','conc']).agg({"s_index":["count","mean","median","min","max","std","var"]})
        st.write(df_cal) 
        df['group'] = df['specimen'].str.cat(df['conc'],sep='')
        
        # replace the labels in the 'stage' column
        df['stage'] = df['stage'].replace({'stage_I': 'I', 'stage_II': 'II', 'stage_III': 'III', 'stage_IV': 'IV'})
    
        # sort the DataFrame by the 'cancer_name' column
        df = df.sort_values(by='cancer_name')
        # sort the DataFrame by the 'conc' column
        df = df.sort_values(by='conc')
        
        # create a density heatmap with facets for each cancer type and stage
        fig = px.density_heatmap(df, x='cancer_name', y='conc', z='s_index', histfunc='avg',
                                color_continuous_scale='RdBu', range_color=[-0.25, 0.25],
                                facet_row='stage', title='S-Index by Cancer Type and Stage', category_orders={'stage':['I','II','III','IV']}, text_auto=True, height= 600)
        

        # show the plot
        st.plotly_chart(fig)
        
  