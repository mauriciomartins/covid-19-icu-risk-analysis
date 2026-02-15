"""
COVID-19 ICU Risk Analysis
Author: Mauricio Martins

This script performs exploratory analysis and visualization
of COVID-19 ICU admission data from the Kaggle Sírio-Libanês dataset.
"""

import re
import requests
import urllib3
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =====================================================
# DATA LOADING
# =====================================================

DATA_URL = "https://raw.githubusercontent.com/mauriciomartins/covid-19-icu-risk-analysis/main/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx"


def load_data(url: str) -> pd.DataFrame:
    """
    Downloads and loads the Excel dataset from a remote URL.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return pd.read_excel(BytesIO(response.content))
    except Exception:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(url, verify=False, timeout=30)
        response.raise_for_status()
        return pd.read_excel(BytesIO(response.content))


# =====================================================
# DATA CLEANING & FEATURE ENGINEERING
# =====================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and prepares dataset:
    - Removes redundant statistical columns
    - Creates age_group and numeric age features
    """

    df = df.copy()

    # Remove redundant statistical aggregations
    remove_patterns = ("_DIFF", "_MIN", "_MAX", "_MEDIAN")
    remove_cols = [col for col in df.columns if col.endswith(remove_patterns)]

    df.drop(columns=remove_cols, errors="ignore", inplace=True)

    # Create age_group
    df["age_group"] = df["AGE_ABOVE65"].map({1: "+65", 0: "<65"})

    # Extract numeric age from AGE_PERCENTIL
    df["age"] = df["AGE_PERCENTIL"].apply(
        lambda x: float(re.sub(r"[^\d.]", "", str(x)))
    )

    return df


# =====================================================
# PATIENT-LEVEL AGGREGATION
# =====================================================

def aggregate_patient_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses dataset to one row per patient
    to avoid window duplication bias.
    """
    return (
        df.groupby("PATIENT_VISIT_IDENTIFIER")
        .agg({
            "ICU": "max",
            "age_group": "first",
            "age": "first"
        })
        .reset_index()
    )


# =====================================================
# VISUALIZATIONS
# =====================================================

def plot_dashboard(df: pd.DataFrame):
    """
    Creates ICU analysis dashboard with multiple subplots.
    """

    patient_df = aggregate_patient_level(df)

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "ICU by Age Group",
            "ICU by Age",
            "ICU Rate by Time Window",
            "ICU Frequency by Time Window"
        )
    )

    # ---------------------------------------------
    # ICU by Age Group
    # ---------------------------------------------
    freq_age_group = pd.crosstab(
        patient_df["age_group"],
        patient_df["ICU"]
    ).reset_index()

    fig.add_trace(
        go.Bar(
            x=freq_age_group["age_group"],
            y=freq_age_group[0],
            name="ICU = No",
            text=freq_age_group[0],
            textposition="auto" 
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=freq_age_group["age_group"],
            y=freq_age_group[1],
            name="ICU = Yes",
            text=freq_age_group[1],
            textposition="auto"
        ),
        row=1, col=1
    )

    # ---------------------------------------------
    # ICU by Age (numeric)
    # ---------------------------------------------
    freq_age = pd.crosstab(
        patient_df["age"],
        patient_df["ICU"]
    ).reset_index()

    fig.add_trace(
        go.Bar(
            x=freq_age["age"],
            y=freq_age[1],
            name="ICU Yes by Age",
            text=freq_age[1],
            textposition="auto"
        ),
        row=2, col=1
    )

    # ---------------------------------------------
    # ICU Rate by Window
    # ---------------------------------------------
    icu_by_window = df.groupby("WINDOW")["ICU"].mean().reset_index()

    fig.add_trace(
        go.Scatter(
            x=icu_by_window["WINDOW"],
            y=icu_by_window["ICU"],
            mode="lines+markers",
            name="ICU Rate",
            text=icu_by_window["ICU"].apply(lambda x: f"{x:.2%}")
        ),
        row=3, col=1
    )

    # ---------------------------------------------
    # ICU Frequency by Window
    # ---------------------------------------------
    freq_window = pd.crosstab(
        df["WINDOW"],
        df["ICU"]
    ).reset_index()

    fig.add_trace(
        go.Bar(
            x=freq_window["WINDOW"],
            y=freq_window[1],
            name="ICU Yes",
            text=freq_window[1],
            textposition="auto"
        ),
        row=4, col=1
    )

    fig.update_layout(
        height=1000,
        title="COVID-19 ICU Analysis Dashboard",
        barmode="group"
    )

    fig.show()