import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
import time
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Path to your Streamlit script
STREAMLIT_SCRIPT_PATH = "app.py"

@pytest.fixture
def app():
    return AppTest.from_file(STREAMLIT_SCRIPT_PATH)

# Test 1: App Title
def test_app_title(app):
    """Test if the app title is correctly displayed."""
    app.run()
    assert "Customer Behavior Prediction" in app.title[0].value

# Test 2: Sidebar Inputs
def test_sidebar_inputs(app):
    """Test if all sidebar inputs are present and have correct labels and types."""
    app.run()
    
    input_checks = [
        ("Tenure (months)", "number_input"),
        ("Preferred Login Device", "selectbox"),
        ("City Tier", "selectbox"),
        ("Warehouse To Home Distance", "number_input"),
        ("Preferred Payment Mode", "selectbox"),
        ("Gender", "selectbox"),
        ("Hours Spent On App", "number_input"),
        ("Number Of Devices Registered", "number_input"),
        ("Preferred Order Category", "selectbox"),
        ("Satisfaction Score", "number_input"),
        ("Marital Status", "selectbox"),
        ("Number Of Addresses", "number_input"),
        ("Complain", "selectbox"),
        ("Order Amount Hike From Last Year (%)", "number_input"),
        ("Coupon Used", "number_input"),
        ("Order Count", "number_input"),
        ("Days Since Last Order", "number_input"),
        ("Cashback Amount", "number_input")
    ]
    
    for label, input_type in input_checks:
        input_element = getattr(app.sidebar, input_type)
        assert any(widget.label == label for widget in input_element)

# Test 3: Prediction Button
def test_prediction_button(app):
    """Test if the prediction button is present and functional."""
    app.run()
    assert "Predict" in app.button[0].label
    
    app.button[0].click().run()
    assert any("Prediction:" in header.value for header in app.header)

# Test 4: Input Validation
def test_input_validation(app):
    """Test input validation for numeric fields."""
    app.run()
    
    numeric_inputs = [
        ("Tenure (months)", 0),
        ("Warehouse To Home Distance", 0),
        ("Hours Spent On App", 0),
        ("Number Of Devices Registered", 0),
        ("Satisfaction Score", 0),
        ("Number Of Addresses", 0),
        ("Order Amount Hike From Last Year (%)", 0),
        ("Coupon Used", 0),
        ("Order Count", 0),
        ("Days Since Last Order", 0),
        ("Cashback Amount", 0)
    ]
    
    for label, min_value in numeric_inputs:
        input_widget = next(widget for widget in app.sidebar.number_input if widget.label == label)
        input_widget.set_value(-1).run()
        assert input_widget.value == min_value

# Test 5: Model and Scaler Loading
def test_model_loading():
    """Test if the model and scaler can be loaded successfully."""
    with open('decision_tree_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    assert model is not None
    assert scaler is not None

# Test 6: Prediction Performance
def test_prediction_performance():
    """Test the performance of making a prediction."""
    with open('decision_tree_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    sample_input = pd.DataFrame({
        'Tenure': [12],
        'CityTier': [1],
        'WarehouseToHome': [10],
        'HourSpendOnApp': [2],
        'NumberOfDeviceRegistered': [1],
        'SatisfactionScore': [4],
        'NumberOfAddress': [2],
        'Complain': [0],
        'OrderAmountHikeFromlastYear': [10],
        'CouponUsed': [2],
        'OrderCount': [5],
        'DaySinceLastOrder': [3],
        'CashbackAmount': [100],
        'PreferredLoginDevice_Computer': [1],
        'PreferredLoginDevice_Mobile Phone': [0],
        'PreferredLoginDevice_Phone': [0],
        'PreferredPaymentMode_CC': [0],
        'PreferredPaymentMode_COD': [0],
        'PreferredPaymentMode_Cash on Delivery': [1],
        'PreferredPaymentMode_Credit Card': [0],
        'PreferredPaymentMode_Debit Card': [0],
        'PreferredPaymentMode_E wallet': [0],
        'PreferredPaymentMode_UPI': [0],
        'Gender_Female': [1],
        'Gender_Male': [0],
        'PreferedOrderCat_Fashion': [0],
        'PreferedOrderCat_Grocery': [1],
        'PreferedOrderCat_Laptop & Accessory': [0],
        'PreferedOrderCat_Mobile': [0],
        'PreferedOrderCat_Mobile Phone': [0],
        'PreferedOrderCat_Others': [0],
        'MaritalStatus_Divorced': [0],
        'MaritalStatus_Married': [1],
        'MaritalStatus_Single': [0]
    })
    
    start_time = time.time()
    input_scaled = scaler.transform(sample_input)
    prediction = model.predict(input_scaled)
    end_time = time.time()
    
    assert (end_time - start_time) < 1  # Assuming prediction should take less than 1 second
    assert prediction in [0, 1]

# Test 7: Error Handling
def test_error_handling(app):
    """Test error handling for invalid inputs."""
    app.run()
    
    with pytest.raises(Exception):  # The exact exception type may vary
        app.sidebar.number_input[0].set_value("invalid").run()

# Test 8: Selectbox Options
def test_selectbox_options(app):
    """Test if selectboxes have the correct options."""
    app.run()
    
    selectbox_options = [
        ("Preferred Login Device", ['Mobile Phone', 'Phone', 'Computer']),
        ("City Tier", [1, 2, 3]),
        ("Preferred Payment Mode", ['Debit Card', 'UPI', 'CC', 'Cash on Delivery', 'E wallet', 'COD', 'Credit Card']),
        ("Gender", ['Male', 'Female']),
        ("Preferred Order Category", ['Laptop & Accessory', 'Mobile', 'Mobile Phone', 'Others', 'Fashion', 'Grocery']),
        ("Marital Status", ['Single', 'Divorced', 'Married']),
        ("Complain", [0, 1])
    ]
    
    for label, options in selectbox_options:
        selectbox = next(widget for widget in app.sidebar.selectbox if widget.label == label)
        assert set(selectbox.options) == set(options)

# Test 9: Input Display
def test_input_display(app):
    """Test if the input features are correctly displayed."""
    app.run()
    
    app.sidebar.number_input[0].set_value(12).run()
    app.sidebar.selectbox[0].select("Mobile Phone").run()
    
    assert "Input Features" in app.subheader[0].value
    assert any("Tenure" in df.value for df in app.dataframe)

# Test 10: Prediction Display
def test_prediction_display(app):
    """Test if the prediction is correctly displayed for both outcomes."""
    app.run()
    
    # Set input values for likely churn
    app.sidebar.number_input[0].set_value(1).run()  # Short tenure
    app.sidebar.selectbox[1].select(3).run()  # Higher city tier
    app.sidebar.number_input[2].set_value(1).run()  # Low app usage
    app.sidebar.number_input[4].set_value(1).run()  # Low satisfaction
    app.sidebar.selectbox[6].select(1).run()  # Has complained
    
    app.button[0].click().run()
    assert any("Prediction: Customer will churn." in header.value for header in app.header)
    
    # Set input values for likely non-churn
    app.sidebar.number_input[0].set_value(24).run()  # Longer tenure
    app.sidebar.selectbox[1].select(1).run()  # Lower city tier
    app.sidebar.number_input[2].set_value(5).run()  # High app usage
    app.sidebar.number_input[4].set_value(5).run()  # High satisfaction
    app.sidebar.selectbox[6].select(0).run()  # No complaints
    
    app.button[0].click().run()
    assert any("Prediction: Customer will not churn." in header.value for header in app.header)

# Test 11: Model Columns Consistency
def test_model_columns_consistency():
    """Test if the model columns match the expected columns."""
    expected_columns = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
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
    
    with open('decision_tree_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    assert set(model.feature_names_in_) == set(expected_columns)

# Test 12: Scaler Functionality
def test_scaler_functionality():
    """Test if the scaler properly transforms the input data."""
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    sample_input = np.array([[12, 1, 10, 2, 1, 4, 2, 0, 10, 2, 5, 3, 100]])
    scaled_input = scaler.transform(sample_input)
    
    assert scaled_input.shape == sample_input.shape
    assert np.all((scaled_input >= -5) & (scaled_input <= 5))  # Assuming standard scaling

# Custom pytest plugin to collect test results
class ReportPlugin:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0
        self.test_results = []

    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            self.total += 1
            if report.outcome == 'passed':
                self.passed += 1
            elif report.outcome == 'failed':
                self.failed += 1
            self.test_results.append((report.nodeid, report.outcome))

def generate_report(plugin):
    """Generate a detailed report of the test results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"Test Report - Generated on {now}\n\n"
    report += f"Total Tests: {plugin.total}\n"
    report += f"Passed: {plugin.passed}\n"
    report += f"Failed: {plugin.failed}\n"
    report += f"Pass Rate: {(plugin.passed / plugin.total) * 100:.2f}%\n\n"
    report += "Detailed Test Results:\n"
    for test, outcome in plugin.test_results:
        report += f"{test}: {outcome}\n"
    return report

if __name__ == "__main__":
    # Create an instance of the custom plugin
    report_plugin = ReportPlugin()
    
    # Run pytest with the custom plugin
    pytest.main([__file__, "-v"], plugins=[report_plugin])
    
    # Generate the report
    report = generate_report(report_plugin)
    
    # Print the report to console
    print(report)
    
    # Save the report to a file
    with open("test_report.txt", "w") as f:
        f.write(report)
    
    print("Test report has been saved to 'test_report.txt'")