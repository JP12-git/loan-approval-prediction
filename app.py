from platform import node
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('Preprocessing and Model.pkl')
feature = joblib.load('Feature.pkl')

st.set_page_config(page_title='Loan Approval Prediction',layout='centered',initial_sidebar_state='expanded')
st.title("🏦 Loan Approval Prediction 💰")
st.divider()

st.subheader(" 📌 About Project")
st.write("""
This Loan Approval Prediction project uses Machine Learning to predict whether a
loan application is likely to be Approved or Rejected based on applicant information.

The project includes data cleaning, missing-value handling, categorical encoding,
exploratory data analysis, model training and model evaluation.
""")

with st.expander(" ⚖️ Decision Tree Model"):

    st.write("""
    A Decision Tree makes predictions by learning a series of decision rules
    from the training data. In this project, the model uses applicant
    information such as income, loan amount, credit score, age and
    categorical details to classify the loan status.
    """)

    st.markdown("**Model Configuration :**")

    st.write("""
    • Algorithm : Decision Tree Classifier  
    • Criterion : Gini  
    • Maximum Depth : 5  
    • Task : Binary Classification  
    • Classes : Approved / Rejected
    """)

with st.expander("📊 Model Performance"):

    st.markdown("### Evaluation Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🎯 Test Accuracy", "97.49%")

    with col2:
        st.metric("📚 Train Score", "97.08%")

    with col3:
        st.metric("🔄 5-Fold CV Score", "96.25%")

    st.markdown("### 📋 Classification Report")

    report_data = pd.DataFrame({
        "Class": ["Approved", "Rejected"],
        "Precision": [0.98, 0.96],
        "Recall": [0.99, 0.92],
        "F1-Score": [0.98, 0.94]
    })

    st.dataframe(
        report_data,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("### 🔲 Confusion Matrix")

    cm_data = pd.DataFrame(
        [[771, 8],
         [17, 199]],
        index=["Actual Approved", "Actual Rejected"],
        columns=["Predicted Approved", "Predicted Rejected"]
    )

    st.dataframe(
        cm_data,
        use_container_width=True
    )

    st.caption(
        "The model correctly classified 771 Approved and 199 Rejected "
        "applications on the test data."
    )

with st.expander("⚙️ How It Works"):

    st.markdown("""
    **1️⃣ Enter Applicant Information**  
    Provide the required applicant and loan details.

    **2️⃣ Preprocessing**  
    Missing values are handled and categorical information is converted into a
    machine-readable format.

    **3️⃣ Prediction**  
    The trained Decision Tree processes the information and predicts the loan status.

    **4️⃣ Result**  
    The application displays the predicted **Loan Approval Status**.
    """)

st.divider()
st.write(' 📝 Enter The Candidate Information Below ⤵️')
st.write("")

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

st.divider()
Input = pd.DataFrame([user])
st.subheader(" 📜 User Input")

df = pd.DataFrame({"Loan Feature":Input.columns,"Entered Values":Input.values[0]})

st.table(df)

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

st.write("")
st.write("")
if st.button("🔮 Predict Loan Status 🕹️",use_container_width=True):

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

    st.warning(
                """⚠️ **Important Note :** This prediction is generated by a machine learning model
                        and may not always be correct.
                        This project is developed for educational and demonstration purposes.
                """)
st.divider()
st.caption(""" 🏦 Loan Approval Prediction | Machine Learning Project | 🧑‍💻 Developed by Prerak Jasani """)