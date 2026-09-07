from platform import node

import streamlit as st
import joblib
import pandas as pd

model = joblib.load('Preprocessing and Model.pkl')
feature = joblib.load('Feature.pkl')

st.set_page_config(page_title='Loan Risk Prediction',layout='centered',initial_sidebar_state='expanded')
st.title("🏦 Loan Approval Prediction 💰")
st.write(' 📝 Enter The Candidate Information Below ⤵️')

numeric_col = ['Age','Income','LoanAmount','CreditScore']

selectbox_col = {
    'Education':['High School','Bachelors','Masters','PhD'],
    'Gender':['Male','Female'],
    'EmploymentType':['Self-Employed','Salaried','Unemployed']
}

user = {}

for col in feature :

    if col in numeric_col :

        user[col] = st.number_input(col,value=None,placeholder='Enter Here')

    elif col in selectbox_col :

        user[col] = st.selectbox('Select Value',selectbox_col[col]) 

    else :

        user[col] = st.text_input(f'Enter {col}')

Input = pd.DataFrame([user])
st.subheader(" 🗒️ User Input")
st.dataframe(Input,hide_index=True)



def get_decision_reasons(model, Input):

    prep = model.named_steps['Preprocessing']
    tree = model.named_steps['Model']

    # Transform input using preprocessing pipeline
    x_transformed = prep.transform(Input)

    # Feature names after preprocessing
    features_name = prep.get_feature_names_out()

    # Get decision path
    node_indicator = tree.decision_path(x_transformed)

    node_index = node_indicator.indices[
        node_indicator.indptr[0]:
        node_indicator.indptr[1]
    ]

    reasons = []

    for node_id in node_index[:-1]:

        feature_index = tree.tree_.feature[node_id]
        threshold = tree.tree_.threshold[node_id]

        # Skip leaf node
        if feature_index == -2:
            continue

        feature_name = features_name[feature_index]

        # --------------------------------------------------
        # Only Numeric Features
        # --------------------------------------------------

        if not feature_name.startswith("Numeric__"):
            continue

        # Convert Numeric__CreditScore → CreditScore
        clean_name = feature_name.replace(
            "Numeric__",
            "",
            1
        )
        
        user_value = Input[clean_name].iloc[0]

        # Determine direction
        if user_value <= threshold:

            direction = "<="

        else:

            direction = ">"

        # Add reason
        reasons.append(
            f"{clean_name} {direction} {threshold:.2f}"
        )

    return reasons


if st.button("🔮 Predict Loan Status 🕹️"):

    if Input.isnull().any().any():

        st.warning("⚠️ Please Enter All Required Information")

    else:

        Pred = model.predict(Input)

        st.subheader("🎯 Result")

        if Pred[0] == "Approved":

            st.success("🟩 ✅ Loan Approved")

        elif Pred[0] == "Rejected":

            st.error("🟥 ❌ Loan Rejected")

            st.subheader("🔍 Why Was The Loan Rejected ?")
            st.write(
                    "According to the Decision Tree model , "
                    "the prediction followed these decision rules :"
                )
            reasons = get_decision_reasons(model, Input)
            for reason in reasons:
                    st.warning("• " + reason)

st.markdown("""<div style ="text-align: right; 
font-family: Constantia; 
font-style: italic; 
font-weight: bold ; 
font-size: 20px; 
color:gray ;">
Developed by Prerak Jasani 
</div>""", unsafe_allow_html=True)
