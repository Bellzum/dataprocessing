import streamlit as st
import pandas as pd
import openpyxl
import plotly.express as px
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from sklearn.decomposition import PCA

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

    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)

        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")

            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]

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
                df = df[df[column].between(*user_num_input)]

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
                    df = df.loc[df[column].between(start_date, end_date)]

            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

def run(key):
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
        sheet_selector = st.sidebar.selectbox("Select sheet:", wb.sheetnames)
        df = pd.read_excel(data_file, sheet_selector)
        st.markdown(f"### Currently Selected worksheet: `{sheet_selector}`")

        st.dataframe(filter_dataframe(df))

        mean_result = st.button("Analyze data...")
        if mean_result:
            st.header('Evaluation of data')
            df_mean = df.groupby(['specimen']).agg({"s_index":["mean"]})
            df_cal = df.groupby(['specimen', 'conc']).agg({"s_index":["count","mean","median","min","max","std","var"]})
            st.write(df_cal) 

            df['group'] = df['specimen'].str.cat(df['conc'], sep='')

            st.subheader('p-value')
            st.caption('***Tukey’s honestly significant difference (HSD) test performs pairwise comparison of means for a set of samples. Whereas ANOVA (e.g. f_oneway) assesses whether the true means underlying each sample are identical, Tukey’s HSD is a post hoc test used to compare the mean of each sample to the mean of each other sample.***')
            
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            from statsmodels.stats.multicomp import MultiComparison

            mc = MultiComparison(df['s_index'], df['group'])
            result = mc.tukeyhsd()
            st.code(result)
            st.caption('*In this table, the "p-adj" column means p-value.')
            st.caption('*The "reject" column shows that true is "Significant difference".')

            st.header('Bar plot')
            kd1 = df.iloc[1, 1]
            kd2 = df.iloc[1, 5]
            st.subheader('nematode: {}, type: {}'.format(kd1, kd2))
            
            agg_df = df.groupby(['specimen', 'conc'], as_index=False).agg({'s_index':['mean', 'std']})
            agg_df.columns = ['specimen', 'conc', 'mean', 'sd']
            agg_df['SE'] = agg_df['sd'] / np.sqrt(df.groupby(['specimen', 'conc']).size().values)

            fig = px.bar(agg_df, x=('specimen'), y='mean', error_y='SE', color='conc', barmode='group', range_y=(-0.20, 0.20), width=600, height=400, color_discrete_sequence=px.colors.qualitative.Pastel2).update_layout(
                xaxis_title='Cancer type', yaxis_title='Index(-)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig, use_container_width=True)

            st.header('Violin and Scatter plot')

            fig2 = px.violin(df, x='specimen', y='s_index', box=True, points='all', color='conc', violinmode='group', range_y=(-0.75, 0.75), width=600, height=400, color_discrete_sequence=px.colors.qualitative.Pastel2).update_layout(
                xaxis_title='Cancer type', yaxis_title='Index(-)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            
            st.plotly_chart(fig2, use_container_width=True)

            st.header('PCA plot and 3D plot')

            df_pivot = pd.pivot_table(df, values='s_index', index='conc', columns='specimen')
            
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(df_pivot.values)
            df_pc = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            df_pc['specimen'] = df_pivot.index

            st.subheader('PCA plot average')
            fig_pca = px.scatter(df_pc, x='PC1', y='PC2', color='specimen', title='PCA Plot')
            
            st.plotly_chart(fig_pca)

            st.subheader('3D plot average')

            x_vals = df['specimen'].unique()
            y_vals = df['conc'].unique()
            z_vals = df_pivot.values

            fig_surface = go.Figure(data=[go.Surface(z=z_vals, x=x_vals, y=y_vals)])
            fig_surface.update_layout(title='3D Surface Plot')
            st.plotly_chart(fig_surface)

            st.subheader('PCA plot')

            features = ['specimen', 'conc', 's_index']   

            X = df[features]
            y = df['s_index']

            X_enc = pd.get_dummies(X, columns=['specimen', 'conc'])

            X_norm = (X_enc - X_enc.mean()) / X_enc.std()

            pca = PCA(n_components=3)
            pca.fit(X_norm)

            transformed_data = pca.transform(X_norm)

            pc_cols = ['PC1', 'PC2', 'PC3']
            pca_df = pd.DataFrame(transformed_data, columns=pc_cols)
            pca_df['specimen'] = X['specimen']
            pca_df['conc'] = X['conc']

            fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='specimen',
                                symbol='conc', hover_name='specimen',
                                hover_data={'conc': True, 'PC1': False, 'PC2': False, 'PC3': False})
            st.plotly_chart(fig)

            subset_df = df.loc[df['conc'].isin(['1_10(1)', '1_10(2)', '1_10(3)','1_10(4)'])]

            fig = px.scatter_3d(subset_df, x='conc', y='specimen', z='s_index', color='conc', symbol='conc')
            fig.update_layout(scene=dict(xaxis_title='conc', yaxis_title='specimen', zaxis_title='s_index'))

            st.plotly_chart(fig)

if __name__ == "__main__":
    run(key="Chemotaxis_assay_index")
