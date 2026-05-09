import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ----- Train Model -----
df = pd.read_csv("StudentsPerformance.csv")
df["pass_fail"] = (df["math score"] >= 50).astype(int)

encoders = {}
cols = ["gender", "race/ethnicity", "parental level of education",
        "lunch", "test preparation course"]

for col in cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop(["math score", "pass_fail"], axis=1)
y = df["pass_fail"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----- UI -----
st.title("Student Performance Predictor")
st.write("Fill in the student details to predict if they will pass math.")

gender = st.selectbox("Gender", ["female", "male"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.slider("Reading Score", 0, 100, 50)
writing_score = st.slider("Writing Score", 0, 100, 50)

if st.button("Predict"):
    input_data = pd.DataFrame([[
        encoders["gender"].transform([gender])[0],
        encoders["race/ethnicity"].transform([race])[0],
        encoders["parental level of education"].transform([parent_edu])[0],
        encoders["lunch"].transform([lunch])[0],
        encoders["test preparation course"].transform([test_prep])[0],
        reading_score,
        writing_score
    ]], columns=X.columns)

    result = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][result]

    if result == 1:
        st.success(f"✅ Predicted: PASS — {probability*100:.1f}% confidence")
    else:
        st.error(f"❌ Predicted: FAIL — {probability*100:.1f}% confidence")

