import streamlit as st
import pandas as pd
import pickle

# Load the saved model and scaler
model_path = 'decision_tree_model.pkl'
scaler_path = 'scaler.pkl'

# Load the trained model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the scaler used during training
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Define the title for the web app
st.title("Customer Behavior Prediction")

# Sidebar section for user input
st.sidebar.header("Input Features")

# Function to gather user input through sidebar widgets
def user_input_features():
    Tenure = st.sidebar.number_input('Tenure (months)', min_value=0)
    PreferredLoginDevice = st.sidebar.selectbox('Preferred Login Device', ['Mobile Phone', 'Phone', 'Computer'])
    CityTier = st.sidebar.selectbox('City Tier', [1, 2, 3])
    WarehouseToHome = st.sidebar.number_input('Warehouse To Home Distance', min_value=0)
    PreferredPaymentMode = st.sidebar.selectbox('Preferred Payment Mode', ['Debit Card', 'UPI', 'CC', 'Cash on Delivery', 'E wallet', 'COD', 'Credit Card'])
    Gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    HourSpendOnApp = st.sidebar.number_input('Hours Spent On App', min_value=0)
    NumberOfDeviceRegistered = st.sidebar.number_input('Number Of Devices Registered', min_value=0)
    PreferedOrderCat = st.sidebar.selectbox('Preferred Order Category', ['Laptop & Accessory', 'Mobile', 'Mobile Phone', 'Others', 'Fashion', 'Grocery'])
    SatisfactionScore = st.sidebar.number_input('Satisfaction Score', min_value=0)
    MaritalStatus = st.sidebar.selectbox('Marital Status', ['Single', 'Divorced', 'Married'])
    NumberOfAddress = st.sidebar.number_input('Number Of Addresses', min_value=0)
    Complain = st.sidebar.selectbox('Complain', [0, 1])
    OrderAmountHikeFromlastYear = st.sidebar.number_input('Order Amount Hike From Last Year (%)', min_value=0)
    CouponUsed = st.sidebar.number_input('Coupon Used', min_value=0)
    OrderCount = st.sidebar.number_input('Order Count', min_value=0)
    DaySinceLastOrder = st.sidebar.number_input('Days Since Last Order', min_value=0)
    CashbackAmount = st.sidebar.number_input('Cashback Amount', min_value=0)
    
    # Collecting user inputs into a dictionary
    data = {
        'Tenure': Tenure,
        'PreferredLoginDevice': PreferredLoginDevice,
        'CityTier': CityTier,
        'WarehouseToHome': WarehouseToHome,
        'PreferredPaymentMode': PreferredPaymentMode,
        'Gender': Gender,
        'HourSpendOnApp': HourSpendOnApp,
        'NumberOfDeviceRegistered': NumberOfDeviceRegistered,
        'PreferedOrderCat': PreferedOrderCat,
        'SatisfactionScore': SatisfactionScore,
        'MaritalStatus': MaritalStatus,
        'NumberOfAddress': NumberOfAddress,
        'Complain': Complain,
        'OrderAmountHikeFromlastYear': OrderAmountHikeFromlastYear,
        'CouponUsed': CouponUsed,
        'OrderCount': OrderCount,
        'DaySinceLastOrder': DaySinceLastOrder,
        'CashbackAmount': CashbackAmount
    }
    
    # Creating a DataFrame from the user input for display
    features = pd.DataFrame(data, index=[0])
    return features

# Calling the function to gather user input
input_df = user_input_features()

# Displaying the input features collected from the user
st.subheader('Input Features')
st.write(input_df)

# Encoding categorical variables for modeling
input_df_encoded = pd.get_dummies(input_df)

# List of columns expected by the model after encoding
model_columns = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
       'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
       'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
       'DaySinceLastOrder', 'CashbackAmount', 'PreferredLoginDevice_Computer',
       'PreferredLoginDevice_Mobile Phone', 'PreferredLoginDevice_Phone',
       'PreferredPaymentMode_CC', 'PreferredPaymentMode_COD',
       'PreferredPaymentMode_Cash on Delivery',
       'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card',
       'PreferredPaymentMode_E wallet', 'PreferredPaymentMode_UPI',
       'Gender_Female', 'Gender_Male', 'PreferedOrderCat_Fashion',
       'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop & Accessory',
       'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile Phone',
       'PreferedOrderCat_Others', 'MaritalStatus_Divorced',
       'MaritalStatus_Married', 'MaritalStatus_Single']

# Ensure all expected columns are present even if absent in input
for col in model_columns:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

# Reorder columns to match the model's expectations
input_df_encoded = input_df_encoded[model_columns]

# Button to trigger prediction
if st.button('Predict'):
    # Scale the input data using the previously fitted scaler
    input_data_scaled = scaler.transform(input_df_encoded)
    
    # Make predictions using the loaded model
    prediction = model.predict(input_data_scaled)
    
    # Displaying prediction results based on model output
    st.subheader('Prediction')
    if prediction[0] == 0:
        st.header("Prediction: Customer will not churn.")
    elif prediction[0] == 1:
        st.header("Prediction: Customer will churn.")
