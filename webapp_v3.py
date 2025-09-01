import streamlit as st # makes the website app
import pandas as pd # helpd read and work with excel tables
import openpyxl # reads excel files
import plotly.express as px # make colorful graphs
import numpy as np # helps with numbers and maths
import plotly.graph_objects as go
from sklearn.decomposition import PCA # helps with smart analysis (such as PCA)
from pandas.api.types import is_object_dtype, is_datetime64_any_dtype, is_categorical_dtype, is_numeric_dtype

# function to add filters to data
## input: table (called df), output: a filtered table
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args: # the *arg (tuple) means passing an arbitrary number of arguments and
    **kwargs (dict) means passing an arbitrary number of arguments with keywords.
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    # this adds a checkbox.
    modify = st.checkbox("Add filters")

    if not modify: #if not check it, it just shows the full table.
        return df  #if do check it, the filter options will appear.

    df = df.copy() # make a copy of the table to avoid accidentally changing the original table.

    # go through each column (like 'specimen', 'conc', 's_index')
    for col in df.columns: # loop-through each column
        #if the column is a text, it tries to change it into dates
        if is_object_dtype(df[col]): # if a column looks like text, we try to see if it's date(like"2024-05-01")
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception: #if it fails, we skip it (don't break a app)
                pass
        # remove timezone, just a cleanup step, removes timezone info from date columns
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    #Build filter UI by creating container for filter UI, this creates a place to put the filters.
    modification_container = st.container()

    with modification_container:
        # pick column you want to filter
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)

        # filter logic for each column
        ## split the screen
        for column in to_filter_columns:
            left, right = st.columns((1, 20)) # left: a little arrow, right: the filter UI
            left.write("↳")

            #for category-type or small lists
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                # is_categorical_dtype(array or dtype) return boolean
                # .nunique is count number of distinct elements in specified axis return series with number of distinct elements and ignore NaN
                # column: specimen, values: lung, colon, stomach
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                '''
                right.multiselect: creates a multiselect dropdown inside the right column layout from st.column(...)
                f"Values for {colunm}: the lable shown in the dropdown (e.g. value for cancer type)
                df[column].unique(): the selectable options- all unique values from the specified column.
                default = list(df[column].unique()): all options are pre=selected by default
                (i.e., no filtering happens until the user changes the selection)
                user_cat_input: the output list of selected values, which you can then use to filter the dataframe
                If column = "cancer_type" and values are ["lung", "breast", "colorectal"], the widget will:
                        - Display a multiselect with these 3 values.
                        - Have all of them selected by default.
                        - Let the user filter to, say, just "lung" and "breast".
                '''
                # keeps only the selected values
                df = df[df[column].isin(user_cat_input)]

            # for numeric data: (column:s_index)
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                #slider for min to max value, keeps only rows between those numbers
                df = df[df[column].between(*user_num_input)]
            #for date columns: column: collected_date
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    # see a calendar to pick a start and end date/ keeps only rows within the date range
                    df = df.loc[df[column].between(start_date, end_date)]

            #for text columns(fallback)
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input: # you can type a word or part of a word
                    df = df[df[column].str.contains(user_text_input)]
                #keep only rows that contain the text you typed.
    return df # return filtered table


# calculate PCA loadings
def calculate_pca_loadings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate PCA and return loadings
    Args:
        df (pd.DataFrame): Dataframe containing the data for PCA
    Returns:
        pd.DataFrame: DataFrame containing PCA loadings
    """
    features = ['specimen', 'conc', 's_index']
    X = df[features]
    y = df['s_index']

    X_enc = pd.get_dummies(X, columns=['specimen', 'conc'])
    X_norm = (X_enc - X_enc.mean()) / X_enc.std()

    pca = PCA(n_components=3)
    pca.fit(X_norm)

    loadings_df = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    loadings_df.index = X_enc.columns

    return loadings_df

# the main program
def run(key):

    # upload a file (shows file name, file size, sheet names inside it)
    st.sidebar.title("Upload File")
    data_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx'], key=key)

    if data_file:
        file_details = {
            "Filename": data_file.name,
            "FileType": data_file.type,
            "FileSize": data_file.size
        }
        wb = openpyxl.load_workbook(data_file)

        st.sidebar.subheader("File details:")
        st.sidebar.json(file_details, expanded=False)
        st.sidebar.markdown("----")

        st.header("Sheet view")
        #show the excel sheet
        sheet_selector = st.sidebar.selectbox("Select sheet:", wb.sheetnames)
        df = pd.read_excel(data_file, sheet_selector)
        st.markdown(f"### Currently Selected worksheet: `{sheet_selector}`")

        #the user filter the data (pick certain dates, values, look at ranges)
        st.dataframe(filter_dataframe(df))

        mean_result = st.button("Analyze data...")
        # analyze the data (how many samples for each cancer type, avg, min, max values)
        if mean_result:
            st.header('Evaluation of data')
            df_mean = df.groupby(['specimen']).agg({"s_index":["mean"]})
            df_cal = df.groupby(['specimen', 'conc']).agg({"s_index":["count","mean","median","min","max","std","var"]})
            st.write(df_cal)

            df['group'] = df['specimen'].str.cat(df['conc'], sep='')

            st.header('Bar plot')
            selected_nematode = df.iloc[1, 1]
            selected_type = df.iloc[1, 5]
            st.subheader('Nematode: {}, Type: {}'.format(selected_nematode, selected_type))



            agg_df = df.groupby(['specimen', 'conc'], as_index=False).agg({'s_index':['mean', 'std']})
            agg_df.columns = ['specimen', 'conc', 'mean', 'sd']
            agg_df['SE'] = agg_df['sd'] / np.sqrt(df.groupby(['specimen', 'conc']).size().values)

            #show graphs
            fig = px.bar(agg_df, x='specimen', y='mean', error_y='SE', color='conc', barmode='group', range_y=(-0.20, 0.20), width=600, height=400, color_discrete_sequence=px.colors.sequential.Blues[2:]).update_layout(
                xaxis_title='Cancer type', yaxis_title='Index(-)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig, use_container_width=True)

            st.header('Violin and Scatter plot')

            fig2 = px.line(df, x='conc', y='s_index', color='conc', markers=True, range_y=(-0.75, 0.75), width=600, height=400, color_discrete_sequence=px.colors.sequential.Blues[2:]).update_layout(
                xaxis_title='Cancer type', yaxis_title='Index(-)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)
