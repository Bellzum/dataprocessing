import streamlit as st
import pandas as pd
import openpyxl
import plotly.graph_objects as go
import numpy as np
from pandas.api.types import is_object_dtype, is_datetime64_any_dtype, is_categorical_dtype, is_numeric_dtype

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

            # Analyze data
            mean_result = st.button("Analyze data...")
            if mean_result:
                st.header("Evaluation of Data")
                # Group and calculate statistics
                df_cal = df.groupby(['specimen', 'conc']).agg({
                    "s_index": ["count", "mean", "median", "min", "max", "std", "var"]
                })
                st.write(df_cal)

                # Prepare aggregated data for plotting
                agg_df = df.groupby(['specimen', 'conc'], as_index=False).agg({
                    's_index': ['mean', 'std']
                })
                agg_df.columns = ['specimen', 'conc', 'mean', 'sd']
                agg_df['SE'] = agg_df['sd'] / np.sqrt(df.groupby(['specimen', 'conc']).size().values)

                # Sort concentrations for proper plotting
                agg_df['conc'] = pd.Categorical(agg_df['conc'], ordered=True, categories=sorted(agg_df['conc'].unique()))
                agg_df = agg_df.sort_values(by='conc')

                st.header("Line and Scatter Plots: Grouped Traces")

                # Use string matching to filter relevant specimens
                groups_to_match = ["SH-SY5Y", "diff_med"]
                filtered_agg_df = agg_df[agg_df['specimen'].str.contains('|'.join(groups_to_match), case=False, na=False)]
                filtered_raw_df = df[df['specimen'].str.contains('|'.join(groups_to_match), case=False, na=False)]

                # Check if filtered data is empty
                if filtered_agg_df.empty or filtered_raw_df.empty:
                    st.warning(f"No data available for the selected groups: {groups_to_match}")
                    st.stop()  # Stop execution if no data is available

                # Define colors for each group
                group_colors = {
                    "SH-SY5Y": "cornflowerBlue",
                    "diff_med": "MediumPurple"
                }

                # Create the figure
                fig2 = go.Figure()

                # Add traces for line plot (aggregated data)
                for group in groups_to_match:
                    subset = filtered_agg_df[filtered_agg_df['specimen'].str.contains(group, case=False, na=False)]
                    fig2.add_trace(go.Scatter(
                        x=subset['conc'],  # Concentrations
                        y=subset['mean'],  # Mean s_index
                        mode='lines+markers',
                        name=f'Line (Mean): {group}',
                        line=dict(color=group_colors[group]),
                        marker=dict(color=group_colors[group]),
                        error_y=dict(
                            type='data',
                            array=subset['SE'],  # Standard Error
                            visible=True
                        )
                    ))

                # Add traces for scatter plot (raw data)
                for group in groups_to_match:
                    subset = filtered_raw_df[filtered_raw_df['specimen'].str.contains(group, case=False, na=False)]
                    fig2.add_trace(go.Scatter(
                        x=subset['conc'],  # Concentrations
                        y=subset['s_index'],  # Raw s_index
                        mode='markers',
                        name=f'Scatter (Raw): {group}',
                        marker=dict(size=8, color=group_colors[group], symbol='circle', opacity=0.6),
                    ))

                # Customize the layout
                fig2.update_layout(
                    title="Mean s_index and Raw Data by Concentration for Selected Groups",
                    xaxis_title="Concentration",
                    yaxis_title="s_index",
                    template="plotly_white",
                    width=800,
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    )
                )

                # Display the plot
                st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    run(key="chemotaxis_assay_index")
