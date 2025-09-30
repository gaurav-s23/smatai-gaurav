# app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import random

# st.set_page_config(page_title="Predictive Maintenance Reports", layout="wide")
# st.title("Predictive Maintenance: Short & Long Report Generator")

# # ------------------------------
# # Upload CSV file
# # ------------------------------
# uploaded_file = st.file_uploader("Upload your Predictive Maintenance Dataset (CSV)", type="csv")

# if uploaded_file:
#     data = pd.read_csv(uploaded_file)
#     st.success("Dataset loaded successfully!")
    
#     st.subheader("Raw Data")
#     st.dataframe(data.head())

#     # ------------------------------
#     # Preprocess dataset
#     # ------------------------------
#     non_numeric = data.select_dtypes(exclude=['int64','float64']).columns

#     for col in non_numeric:
#         if col.lower() in ['udi','id','serialnumber']:
#             data = data.drop(col, axis=1)
#         elif col != 'Machine failure':
#             le = LabelEncoder()
#             data[col] = le.fit_transform(data[col])

#     X = data.drop('Machine failure', axis=1)
#     y = data['Machine failure']

#     # ------------------------------
#     # Train Random Forest model
#     # ------------------------------
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)

#     st.subheader("Model Evaluation")
#     st.write("Accuracy:", accuracy_score(y_test, y_pred))
#     st.text(classification_report(y_test, y_pred))

#     # ------------------------------
#     # Generate short and long reports
#     # ------------------------------
#     def generate_short_report(row):
#         lines = []
#         if 'Temperature' in X.columns:
#             lines.append("High temperature detected" if row['Temperature'] > 80 else "Temperature normal")
#         if 'Pressure' in X.columns and row['Pressure'] < 25:
#             lines.append("Low pressure warning")
#         if 'Vibration' in X.columns and row['Vibration'] > 0.03:
#             lines.append("High vibration detected")
#         if 'operator_note' in X.columns and row['operator_note']:
#             lines.append(f"Operator: {row['operator_note']}")
#         return "; ".join(lines) if lines else "All readings normal"

#     def generate_long_report(row):
#         report = f"Maintenance Report for Machine:\nRecent sensor readings:\n"
#         for col in X.columns:
#             if np.issubdtype(type(row[col]), np.number):
#                 report += f" - {col}: {row[col]:.2f}\n"
#             else:
#                 report += f" - {col}: {row[col]}\n"
        
#         report += "\nAnalysis & Recommendations:\n"
#         if 'Temperature' in X.columns:
#             temp = row['Temperature']
#             if temp > 85:
#                 report += "Temperature is critically high. Immediate inspection of cooling system required.\n"
#             elif temp > 75:
#                 report += "Temperature slightly elevated; monitor over next cycles.\n"
#             else:
#                 report += "Temperature is within safe range.\n"
#         if 'Pressure' in X.columns:
#             pressure = row['Pressure']
#             if pressure < 25:
#                 report += "Pressure is low. Check valves and pumps for efficiency.\n"
#             elif pressure > 35:
#                 report += "Pressure is high. Inspect for blockages or leaks.\n"
#             else:
#                 report += "Pressure is normal.\n"
#         if 'Vibration' in X.columns:
#             vib = row['Vibration']
#             if vib > 0.03:
#                 report += "Excessive vibration detected. Inspect bearings and rotating parts.\n"
#             elif vib > 0.02:
#                 report += "Mild vibration observed; monitor mechanical components.\n"
#             else:
#                 report += "Vibration levels normal.\n"
#         if 'operator_note' in X.columns and row['operator_note']:
#             report += f"Operator observations: {row['operator_note']}\n"
#         report += "Overall, maintenance action recommended based on above readings and observations.\n"
#         return report

#     # ------------------------------
#     # Apply to all rows
#     # ------------------------------
#     short_reports = X.apply(generate_short_report, axis=1)
#     long_reports = X.apply(generate_long_report, axis=1)

#     output_df = data.copy()
#     output_df['Short_Report'] = short_reports
#     output_df['Long_Report'] = long_reports

#     # ------------------------------
#     # Display a random report
#     # ------------------------------
#     if st.button("Show Random Report"):
#         random_index = random.randint(0, len(output_df)-1)
#         st.subheader(f"Random Report (Row {random_index})")
#         st.markdown("**Short Report:**")
#         st.write(output_df.loc[random_index, 'Short_Report'])
#         st.markdown("**Long Report:**")
#         st.text(output_df.loc[random_index, 'Long_Report'])


# ==============================
# Predictive Maintenance: Short & Long Report Generator
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import random

st.set_page_config(page_title="Predictive Maintenance Reports", layout="wide")
st.title("Predictive Maintenance: Short & Long Report Generator")

# ------------------------------
# Sidebar: Upload CSV
# ------------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    st.subheader("Raw Data Preview")
    st.dataframe(data.head())

    # ------------------------------
    # Preprocess dataset
    # ------------------------------
    non_numeric = data.select_dtypes(exclude=['int64','float64']).columns

    for col in non_numeric:
        if col.lower() in ['udi','id','serialnumber']:
            data = data.drop(col, axis=1)
        elif col != 'Machine failure':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    X = data.drop('Machine failure', axis=1)
    y = data['Machine failure']

    # ------------------------------
    # Train Random Forest model
    # ------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("Model Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))

    # ------------------------------
    # Generate short and long reports
    # ------------------------------
    def generate_short_report(row):
        lines = []
        if 'Temperature' in X.columns:
            lines.append("High temperature detected" if row['Temperature'] > 80 else "Temperature normal")
        if 'Pressure' in X.columns and row['Pressure'] < 25:
            lines.append("Low pressure warning")
        if 'Vibration' in X.columns and row['Vibration'] > 0.03:
            lines.append("High vibration detected")
        if 'operator_note' in X.columns and row['operator_note']:
            lines.append(f"Operator: {row['operator_note']}")
        return "; ".join(lines) if lines else "All readings normal"

    def generate_long_report(row):
        report = f"Maintenance Report for Machine:\nRecent sensor readings:\n"
        for col in X.columns:
            if np.issubdtype(type(row[col]), np.number):
                report += f" - {col}: {row[col]:.2f}\n"
            else:
                report += f" - {col}: {row[col]}\n"
        report += "\nAnalysis & Recommendations:\n"
        if 'Temperature' in X.columns:
            temp = row['Temperature']
            if temp > 85:
                report += "Temperature is critically high. Immediate inspection of cooling system required.\n"
            elif temp > 75:
                report += "Temperature slightly elevated; monitor over next cycles.\n"
            else:
                report += "Temperature is within safe range.\n"
        if 'Pressure' in X.columns:
            pressure = row['Pressure']
            if pressure < 25:
                report += "Pressure is low. Check valves and pumps for efficiency.\n"
            elif pressure > 35:
                report += "Pressure is high. Inspect for blockages or leaks.\n"
            else:
                report += "Pressure is normal.\n"
        if 'Vibration' in X.columns:
            vib = row['Vibration']
            if vib > 0.03:
                report += "Excessive vibration detected. Inspect bearings and rotating parts.\n"
            elif vib > 0.02:
                report += "Mild vibration observed; monitor mechanical components.\n"
            else:
                report += "Vibration levels normal.\n"
        if 'operator_note' in X.columns and row['operator_note']:
            report += f"Operator observations: {row['operator_note']}\n"
        report += "Overall, maintenance action recommended based on above readings and observations.\n"
        return report

    # Apply to all rows
    short_reports = X.apply(generate_short_report, axis=1)
    long_reports = X.apply(generate_long_report, axis=1)

    output_df = data.copy()
    output_df['Short_Report'] = short_reports
    output_df['Long_Report'] = long_reports

    # ------------------------------
    # Download reports
    # ------------------------------
    csv = output_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Reports as CSV",
        data=csv,
        file_name="predictive_maintenance_reports.csv",
        mime="text/csv"
    )

    # ------------------------------
    # Display a random report
    # ------------------------------
    if st.button("Show Random Report"):
        random_index = random.randint(0, len(output_df)-1)
        st.subheader(f"Random Report (Row {random_index})")
        st.markdown("**Short Report:**")
        st.write(output_df.loc[random_index, 'Short_Report'])
        st.markdown("**Long Report:**")
        st.text_area("Long Report Details", output_df.loc[random_index, 'Long_Report'], height=300)
