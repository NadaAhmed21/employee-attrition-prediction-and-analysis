import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# إعداد الصفحة
st.set_page_config(page_title="Employee Attrition Prediction", page_icon=":briefcase:", layout="wide")

# تخصيص CSS لإخفاء أيقونات الشريط الجانبي وتغيير الخلفية
st.markdown(
    """
    <style>
    /* إخفاء شريط التنقل الجانبي (الأيقونات) */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* إخفاء زر طي الشريط الجانبي */
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* إخفاء شعار Streamlit الافتراضي أو أيقونة الصورة */
    img[alt="App logo"] {
        display: none !important;
    }

    /* تخصيص لون خلفية الشريط الجانبي */
    [data-testid="stSidebar"] {
        background-color: rgb(240, 251, 255);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======== الشريط الجانبي: صورة فقط =========
with st.sidebar:
    try:
        image = Image.open("E.png")  # تأكد من وجود الصورة في نفس مجلد السكربت
        st.image(image, use_container_width=True)
    except FileNotFoundError:
        st.warning("Image 'E.png' not found. Please ensure the file is in the same directory as app.py.")

# ======== المنطقة الرئيسية: العنوان والوصف =========
st.title("Employee Attrition Prediction")
st.markdown("""
This application predicts whether an employee is likely to leave or stay based on their work-related attributes.
Enter the employee's details below to get a prediction using a trained SVM model.
""")

# ======== إدخال بيانات الموظف =========
st.subheader("Enter Employee Details")
number_project = st.slider("Number of Projects", 1, 10, 2)
average_montly_hours = st.slider("Average Monthly Hours", 50, 350, 157)
time_spend_company = st.slider("Time Spent at Company (Years)", 1, 20, 3)
work_accident = st.selectbox("Work Accident", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
department = st.selectbox("Department", ['sales', 'technical', 'support', 'IT', 'RandD', 'product_mng', 'marketing', 'accounting', 'hr', 'management'])
salary = st.selectbox("Salary Level", ['low', 'medium', 'high'])
satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.38, step=0.01)
last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.53, step=0.01)

# تجميع البيانات في DataFrame
input_df = pd.DataFrame({
    'number_project': [number_project],
    'average_montly_hours': [average_montly_hours],
    'time_spend_company': [time_spend_company],
    'Work_accident': [work_accident],
    'promotion_last_5years': [promotion_last_5years],
    'department': [department],
    'salary': [salary],
    'satisfaction_level': [satisfaction_level],
    'last_evaluation': [last_evaluation]
})

# تحميل النموذج
try:
    from sklearn.base import BaseEstimator, TransformerMixin
    import numpy as np

    class OutlierCapper(BaseEstimator, TransformerMixin):
        def __init__(self, factor=1.5):
            self.factor = factor
            self.upper_bounds = {}

        def fit(self, X, y=None):
            for col in X.columns:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                self.upper_bounds[col] = q3 + self.factor * iqr
            return self

        def transform(self, X):
            X = X.copy()
            for col in X.columns:
                if col in self.upper_bounds:
                    X[col] = np.where(X[col] > self.upper_bounds[col], self.upper_bounds[col], X[col])
            return X

    inference_pipeline = joblib.load('svm_inference_pipeline.pkl')
except FileNotFoundError:
    st.error("Model file 'svm_inference_pipeline.pkl' not found.")
    st.stop()

# ======== التنبؤ ========
if st.button("Predict"):
    prediction = inference_pipeline.predict(input_df)
    label = "Stayed" if prediction[0] == 0 else "Left"

    if label == "Stayed":
        st.success(f"Prediction: The employee is likely to **{label}** the company.")
    else:
        st.markdown(
            f"<div style='background-color:#ffe6e6; padding:10px; border-radius:10px;'><strong>Prediction:</strong> The employee is likely to <strong style='color:red;'>Leave</strong> the company.</div>",
            unsafe_allow_html=True
        )
