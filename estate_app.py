import streamlit as st
import pandas as pd
import numpy as np
import sys
import joblib

st.set_page_config(page_title="Gujarat Property Price Predictor", layout="wide")
st.title("Gujarat Real Estate Price Predictor")
st.markdown("**Predict average cost per sq.ft. using My trained model**")

# ================== DEBUG INFO ==================
st.sidebar.header("🔍 Environment Debug")
st.sidebar.write(f"**Python Version:** {sys.version.split()[0]}")
st.sidebar.write(f"**scikit-learn Version:** {sklearn.__version__}")
st.sidebar.write(f"**Joblib Version:** {joblib.__version__}")

try:
    import numpy as np
    st.sidebar.write(f"**NumPy Version:** {np.__version__}")
except:
    st.sidebar.write("NumPy: Not found")
# ===============================================

@st.cache_resource
def load_model():
    model = joblib.load('models/gujarat_rera_gb_model.joblib')
    encoder = joblib.load('models/categorical_ordinal_encoder.joblib')
    return model, encoder

model, encoder = load_model()

# Get exact feature names in correct order from training
feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None

if feature_names is None:
    st.error("Model does not have feature names. Please retrain with newer scikit-learn.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    distName = st.selectbox("District", ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar", "Other"])
    projectType = st.selectbox("Project Type", ["Residential", "Commercial", "Mixed Use", "Other"])
    promoterType = st.selectbox("Promoter Type", ["Individual", "Company", "Other"])
    underRedevelopment = st.selectbox("Under Redevelopment?", ["No", "Yes", "Unknown"])

with col2:
    totalUnits = st.number_input("Total Units", min_value=1, value=100)
    totalAreaOfLand = st.number_input("Total Land Area (sq.ft)", min_value=100.0, value=50000.0,step=1.0)
    totalLandCost = st.number_input("Total Land Cost (₹ Crores)", min_value=0.0, value=5.0)
    totalEstimatedCost = st.number_input("Total Estimated Cost (₹ Crores)", min_value=0.0, value=15.0)
    totalSquareFootBuild = st.number_input("Total Built-up Area (sq.ft)", min_value=100.0, value=80000.0)
    totalProjects = st.number_input("Total Projects", min_value=1, value=1)

# other required fields
totalCarpetArea_form3A = st.number_input("Total Carpet Area (sq.ft)", min_value=100.0, value=60000.0, )
totalBuiltupArea_form3A = st.number_input("Total Built-up Area Form 3A (sq.ft)", min_value=100.0, value=80000.0,step=1.0)
totalDevelopCost = st.number_input("Total Development Cost (₹ Crores)", min_value=0.0, value=12.0)
totalPayableAmountGovernment = st.number_input("Total Payable to Govt (₹ Crores)", min_value=0.0, value=1.5)
totalSellingAmount = st.number_input("Total Selling Amount (₹ Crores)", min_value=0.0, value=25.0)

if st.button("Predict Price per Sq.Ft", type="primary", ):
    input_dict = {
        'distName': [distName],
        'projectType': [projectType],
        'promoterType': [promoterType],
        'underRedevelopment': [underRedevelopment],
        'tpo_code': ["Unknown"],

        'totalUnits': [totalUnits],
        'totalAreaOfLand': [totalAreaOfLand],
        'totalLandCost': [totalLandCost],
        'totalEstimatedCost': [totalEstimatedCost],
        'totalSquareFootBuild': [totalSquareFootBuild],
        'totalProjects': [totalProjects],
        'totalCarpetArea_form3A': [totalCarpetArea_form3A],
        'totalBuiltupArea_form3A': [totalBuiltupArea_form3A],
        'totalDevelopCost': [totalDevelopCost],
        'totalPayableAmountGovernment': [totalPayableAmountGovernment],
        'totalSellingAmount': [totalSellingAmount],


        'AvgAreaOfLand': [totalAreaOfLand / totalProjects],
        'AvgSquareFootBuild': [totalSquareFootBuild / totalUnits if totalUnits > 0 else 0],
        'avgUnits': [totalUnits],
        'EndProjectMonth': [6],
        'EndProjectYear': [2026],
        'avgEstimatedCost_AllProjects': [totalEstimatedCost / totalProjects],
        'startProjectYear': [2024],
        'startProjectMonth': [1],
        'project_duration_days': [730],
        'project_duration_years': [2.0],
    }

    input_df = pd.DataFrame(input_dict)

    cat_cols = ['projectType', 'promoterType', 'underRedevelopment', 'distName', 'tpo_code']
    input_df[cat_cols] = encoder.transform(input_df[cat_cols])

    input_df = input_df[feature_names]

    log_pred = model.predict(input_df)[0]
    predicted_price = np.expm1(log_pred)

    st.success(f"**Predicted Avg Cost: ₹{predicted_price:,.2f} per sq.ft**")

st.caption("Built with Streamlit • And lots of Hardwork")
