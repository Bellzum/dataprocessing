import streamlit as st
import pandas as pd
import openpyxl
import numpy as np
import matplotlib.pyplot as plt

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """ Streamlit filter UI for interactive dataframe filtering. """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()  # Ensure uniform strings

    container = st.container()
    with container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            user_cat_input = st.multiselect(f"Values for {column}", df[column].unique(), default=list(df[column].unique()))
            df = df[df[column].isin(user_cat_input)]

    return df

def run(key):
    st.sidebar.title("Upload File")
    data_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx'], key=key)

    if data_file:
        # Read file and select sheet
        wb = openpyxl.load_workbook(data_file)
        sheet_selector = st.sidebar.selectbox("Select sheet:", wb.sheetnames)
        df = pd.read_excel(data_file, sheet_selector)
        st.markdown(f"### Currently Selected worksheet: `{sheet_selector}`")

        # **Trim whitespaces from specimen & strain**
        df['specimen'] = df['specimen'].astype(str).str.strip()
        df['strain'] = df['strain'].astype(str).str.strip()
        df['conc'] = df['conc'].astype(str).str.strip()

        df_filtered = filter_dataframe(df)
        st.dataframe(df_filtered)

        analyze = st.button("Analyze data...")
        if analyze:
            st.header("Evaluation of Data")

            # Group data by (specimen, strain, conc)
            agg_df = df_filtered.groupby(['specimen','strain','conc'], as_index=False).agg({
                's_index':['mean','std']
            })
            agg_df.columns = ['specimen','strain','conc','mean','sd']

            # Compute standard error
            group_sizes = df_filtered.groupby(['specimen','strain','conc']).size().values
            agg_df['SE'] = agg_df['sd'] / np.sqrt(group_sizes)

            # **Ensure correct specimen ordering & spacing**
            unique_specimens = list(df_filtered['specimen'].unique())  # Maintain original order
            specimen_order = {specimen: i for i, specimen in enumerate(unique_specimens)}

            # **Define the new x-position logic**
            agg_df['specimen_index'] = agg_df['specimen'].map(specimen_order)

            # **Define spacing settings**
            gap_between_specimens = 2.0  # Increase gap between different specimens
            bar_width = 0.6  # Slightly larger bars for better visibility
            shift_within_specimen = 0.1  # Small offset for CR-12 and N2 separation

            # **Create x-axis positions manually**
            x_positions = []
            specimen_strain_labels = []
            last_specimen = None
            x_counter = 0

            for _, row in agg_df.iterrows():
                # If we encounter a new specimen, create a gap
                if last_specimen is not None and last_specimen != row['specimen']:
                    x_counter += gap_between_specimens  # Add spacing for different specimens

                # Adjust within a specimen: CR-12 left, N2 right
                if row['strain'] == "CR-12":
                    x_pos = x_counter - shift_within_specimen
                else:
                    x_pos = x_counter + shift_within_specimen

                x_positions.append(x_pos)
                specimen_strain_labels.append(f"{row['specimen']}_{row['strain']}")

                last_specimen = row['specimen']
                x_counter += 0.7  # Adjust for each concentration shift

            agg_df['x_pos'] = x_positions  # Assign computed x positions

            # **Define colors for strains and conc**
            color_palette = {
                'N2': ["#d9d9d9", "#bdbdbd", "#969696","#636363","#252525","#000000"],  # Grayscale
                'CR-12': ["#deebf7", "#9ecae1", "#6baed6", "#497aab","#08519c","#031066"],  # Blue shades
                'CR-14': ["#fff7ec", "#fee8c8", "#fdd49e", "#fdc086", "#fdae6b", "#f16913"],  # Orange shades
                'CR-17': ["#fde0dd", "#fccde5", "#fa9fb5", "#f768a1", "#dd3497", "#ae017e"],  # Pink pastel shades
                'CR-3':  ["#e0f3ff", "#c2e0f7", "#a3ceef", "#85bbdf", "#68a8d0", "#4b96c0"],  # pastel Blue shade
                'CR-19': ["#f3e5d0", "#e8cdb5", "#ddba99", "#d2a67e", "#c79463", "#bc8249"],  # Pastel brown
                'CR-20': ["#e0f2e9", "#c2e4d3", "#a3d6bd", "#85c9a7", "#66bb91", "#48ad7b"]   # Pastel green
                }

            # **Create the Matplotlib figure**
            fig, ax = plt.subplots(figsize=(18, 6))  # Increase figure size for readability

            # **Plot bars for each (strain, conc) pair separately**
            for (strain, conc), sub_df in agg_df.groupby(['strain', 'conc']):
                strain_colors = color_palette.get(strain, ["#cccccc"])
                strain_concs = sorted(agg_df[agg_df['strain'] == strain]['conc'].unique())
                color_index = strain_concs.index(conc) % len(strain_colors)

                color = strain_colors[color_index]

                ax.bar(
                    sub_df['x_pos'],
                    sub_df['mean'],
                    yerr=sub_df['SE'],
                    color=color,
                    edgecolor="gray", linewidth = 0.5,
                    label=f"{strain} - {conc}" if f"{strain} - {conc}" not in ax.get_legend_handles_labels()[1] else "",
                    error_kw=dict(
                    capsize=2,      # Length of the horizontal cap lines
                    capthick=0.5,     # Thickness of cap lines
                    elinewidth=0.5   # Thickness of error bar lines
                ),
                    width=bar_width
                )

            # **Format the x-axis properly**
            ax.set_xticks(x_positions)
            ax.set_xticklabels(specimen_strain_labels, rotation=20, ha="right", fontsize=5)
            ax.set_xlabel("Specimen + Strain", fontsize=12, fontweight="bold")
            ax.set_ylabel("Chemotaxis Index", fontsize=12, fontweight="bold")
            ax.legend(title="Strain & Concentration", loc="upper right", fontsize=10)

            # **Improve Visual Layout**
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, linestyle="--", alpha=0.3)  # Add light grid for readability
            ax.set_axisbelow(True)  # Ensure gridlines are behind bars

            # **Reduce whitespace between bars**
            plt.tight_layout()

            # **Show plot in Streamlit**
            st.pyplot(fig)

            st.write("**Final x-axis order:**", list(specimen_order.keys()))

if __name__ == "__main__":
    run("ROC_APP")
