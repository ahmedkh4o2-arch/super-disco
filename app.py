import io
import math

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Super Disco Data Analysis", layout="wide")
st.title("Super Disco - Data Analysis App")


def prepare_demo_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    region = np.random.choice(["North", "South", "East", "West"], size=n)
    sales = np.random.normal(5000, 1200, n)
    profit = sales * np.random.uniform(0.12, 0.28, n)
    marketing = np.random.normal(800, 250, n)
    customers = np.random.randint(100, 1000, n)
    days = pd.date_range("2024-01-01", periods=n, freq="D")

    df = pd.DataFrame(
        {
            "date": days,
            "region": region,
            "sales": np.round(sales, 2),
            "profit": np.round(profit, 2),
            "marketing_spend": np.round(marketing, 2),
            "customers": customers,
        }
    )
    return df


@st.cache_data
def load_uploaded_data(file):
    if file is None:
        return prepare_demo_data()

    extension = file.name.lower()
    if extension.endswith(".csv"):
        return pd.read_csv(file)
    if extension.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def show_summary(df: pd.DataFrame):
    st.subheader("Dataset overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", len(df))
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing cells", int(df.isna().sum().sum()))
    col4.metric("Duplicate rows", int(df.duplicated().sum()))

    st.write(df.head())


def show_descriptive_stats(df: pd.DataFrame):
    st.subheader("Summary statistics")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.info("No numeric columns were found for statistical summaries.")
        return
    st.dataframe(numeric_df.describe().T, use_container_width=True)


def show_missing_values(df: pd.DataFrame):
    st.subheader("Missing values")
    missing = df.isna().sum().sort_values(ascending=False)
    if missing.empty or missing.sum() == 0:
        st.success("No missing values found.")
        return

    missing_df = missing[missing > 0].reset_index()
    missing_df.columns = ["Column", "Missing"]
    st.dataframe(missing_df, use_container_width=True)

    fig = px.bar(
        missing_df,
        x="Column",
        y="Missing",
        color="Column",
        title="Missing values by column",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_numeric_charts(df: pd.DataFrame):
    st.subheader("Numeric analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for charts.")
        return

    selected_cols = st.multiselect(
        "Choose numeric columns to visualize",
        numeric_cols,
        default=numeric_cols[: min(3, len(numeric_cols))],
    )

    if not selected_cols:
        st.warning("Please select at least one numeric column.")
        return

    for col in selected_cols:
        st.write(f"### Distribution: {col}")
        fig = px.histogram(df, x=col, nbins=25, title=f"Histogram for {col}")
        st.plotly_chart(fig, use_container_width=True)

    if len(selected_cols) >= 2:
        st.write("### Scatter plot")
        x_col, y_col = st.selectbox("X axis", selected_cols), st.selectbox("Y axis", selected_cols)
        if x_col != y_col:
            fig2 = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig2, use_container_width=True)


def show_correlation(df: pd.DataFrame):
    st.subheader("Correlation analysis")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.info("At least two numeric columns are required to calculate correlations.")
        return

    corr = numeric_df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation heatmap",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Top correlations:")
    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["Column A", "Column B", "Correlation"]
    corr_pairs = corr_pairs[corr_pairs["Correlation"].notna()].sort_values("Correlation", key=abs, ascending=False)
    st.dataframe(corr_pairs.head(10), use_container_width=True)


def generate_insight_summary(df: pd.DataFrame):
    st.subheader("Quick insights")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.info("Add numeric columns to generate insights.")
        return

    summary_lines = []
    summary_lines.append(f"- The dataset contains {len(df)} rows and {df.shape[1]} columns.")

    max_col = numeric_df.max().idxmax()
    max_value = numeric_df.max().max()
    summary_lines.append(f"- The largest value in the numeric data is in '{max_col}' with a value of {max_value:.2f}.")

    missing_counts = df.isna().sum()
    if missing_counts.sum() > 0:
        top_missing = missing_counts.idxmax()
        summary_lines.append(f"- The column with the most missing values is '{top_missing}' with {missing_counts[top_missing]} missing entries.")
    else:
        summary_lines.append("- No missing values were detected across the dataset.")

    corr = numeric_df.corr(numeric_only=True)
    if corr.shape[0] > 1:
        corr_values = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        if not corr_values.empty:
            strongest_pair = corr_values.abs().idxmax()
            strongest_corr = corr_values.loc[strongest_pair]
            summary_lines.append(
                f"- The strongest numeric relationship is between '{strongest_pair[0]}' and '{strongest_pair[1]}' with a correlation of {strongest_corr:.2f}."
            )

    for line in summary_lines:
        st.write(line)


uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

try:
    df = load_uploaded_data(uploaded_file)
except ValueError as e:
    st.error(str(e))
    st.stop()

if uploaded_file is None:
    st.caption("No file uploaded. Showing demo data from a sample sales dataset.")

show_summary(df)
show_descriptive_stats(df)
show_missing_values(df)
show_numeric_charts(df)
show_correlation(df)
generate_insight_summary(df)

st.sidebar.header("About")
st.sidebar.write("This app is a simple starter for analyzing tabular data in Python.")
st.sidebar.write("Upload your own dataset to explore patterns, missing values, and correlations.")
