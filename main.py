import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model from the file
with open('churn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Set page title and icon
st.set_page_config(page_title="Churn Prediction App", page_icon=":bar_chart:")

# Custom styles and layout
st.markdown(
    """
    <style>
    .header {
        color: #ff6347;
        font-size: 36px;
        text-align: center;
        margin-bottom: 30px;
    }
    .subheader {
        color: #4682b4;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .prediction {
        color: #20b2aa;
        font-size: 28px;
        text-align: center;
        margin-top: 30px;
    }
    .footer {
        color: #808080;
        font-size: 14px;
        text-align: center;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title('Churn Prediction App')
st.markdown('<p class="header">Churn Prediction App</p>', unsafe_allow_html=True)

# Sidebar - Input for user features
st.sidebar.header('User Input Features')

def user_input_features():
    tenure = st.sidebar.slider('Tenure (months)', 0, 72, 0)
    phone_service = st.sidebar.radio('Phone Service', ('Yes', 'No'))
    contract = st.sidebar.radio('Contract', ('Month-to-month', 'One year', 'Two year'))
    payment_method = st.sidebar.selectbox('Payment Method', ('N/A', 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    monthly_charges = st.sidebar.slider('Monthly Charges ($)', 0.0, 120.0, 0.0)
    total_charges = st.sidebar.slider('Total Charges ($)', 0.0, 10000.0, 0.0)
    data = {'Tenure (months)': tenure,
            'Phone Service': phone_service,
            'Contract': contract,
            'Payment Method': payment_method,
            'Monthly Charges ($)': monthly_charges,
            'Total Charges ($)': total_charges}
    features = pd.DataFrame(data, index=[0])
    return features

# Display user input features
user_input = user_input_features()
st.markdown('<p class="subheader">User Input Features</p>', unsafe_allow_html=True)
st.write(user_input)

# Make predictions
prediction = model.predict(user_input)

st.markdown('<p class="prediction">Prediction: {}</p>'.format(prediction[0]), unsafe_allow_html=True)

# Correlation analysis
st.subheader('Correlation with Churn')
with st.expander("Click to see correlation heatmap"):
    corr = df.corr()['Churn'].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Footer
st.markdown('<p class="footer">Made with :heart: by Your Name</p>', unsafe_allow_html=True)
