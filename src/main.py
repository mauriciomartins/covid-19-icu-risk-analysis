"""
COVID-19 ICU Risk Analysis
Author: Mauricio Martins

This script performs exploratory analysis and visualization
of COVID-19 ICU admission data from the Kaggle Sírio-Libanês dataset.
"""

from analysis import load_data, preprocess_data, plot_dashboard, DATA_URL

# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    print("Starting data analysis...")

    df = load_data(DATA_URL)
    df = preprocess_data(df)

    plot_dashboard(df)

    print("Analysis completed successfully.")


if __name__ == "__main__":
    main()
