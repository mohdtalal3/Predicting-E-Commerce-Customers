st.title("Customer Behavior Prediction")

def user_input_features():
    # Collect user input here...
    data = {"all features here"}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
st.subheader('Input Features')
st.write(input_df)

input_df_encoded = pd.get_dummies(input_df)
model_columns = ['Tenure', 'CityTier', 'other columns']  

for col in model_columns:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

input_df_encoded = input_df_encoded[model_columns]

if st.button('Predict'):
    input_data_scaled = scaler.transform(input_df_encoded)
    prediction = model.predict(input_data_scaled)
    st.subheader('Prediction')
    st.header("Customer will churn." if prediction[0] == 1 else "Customer will not churn.")