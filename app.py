import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np

st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

MODEL_DIR = Path("models")


@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression.pkl"
        ),
        "Decision Tree": joblib.load(
            MODEL_DIR / "decision_tree.pkl"
        ),
        "Random Forest": joblib.load(
            MODEL_DIR / "random_forest.pkl"
        ),
        "SVM": joblib.load(
            MODEL_DIR / "svm.pkl"
        ),
        "Tuned Random Forest": joblib.load(
            MODEL_DIR / "tuned_random_forest.pkl"
        ),
    }


models = load_models()

st.title("🏨 Hotel Booking Cancellation Predictor")
st.write(
    "Predict whether a hotel booking is likely to be cancelled "
    "using machine learning."
)

st.sidebar.header("Model")
model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)
model = models[model_name]

st.subheader("Booking Information")

col1, col2, col3 = st.columns(3)

with col1:
    hotel = st.selectbox(
        "Hotel",
        ["Resort Hotel", "City Hotel"]
    )

    lead_time = st.number_input(
        "Lead Time",
        min_value=0,
        value=100
    )

    arrival_date_year = st.selectbox(
        "Arrival Year",
        [2015, 2016, 2017]
    )

    arrival_date_month = st.selectbox(
        "Arrival Month",
        [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ]
    )

    arrival_date_week_number = st.number_input(
        "Arrival Week Number",
        min_value=1,
        max_value=53,
        value=20
    )

    arrival_date_day_of_month = st.number_input(
        "Arrival Day",
        min_value=1,
        max_value=31,
        value=15
    )

    stays_in_weekend_nights = st.number_input(
        "Weekend Nights",
        min_value=0,
        value=1
    )

    stays_in_week_nights = st.number_input(
        "Week Nights",
        min_value=0,
        value=2
    )

with col2:
    adults = st.number_input(
        "Adults",
        min_value=1,
        value=2
    )

    children = st.number_input(
        "Children",
        min_value=0,
        value=0
    )

    babies = st.number_input(
        "Babies",
        min_value=0,
        value=0
    )

    meal = st.selectbox(
        "Meal",
        ["BB", "FB", "HB", "SC", "Undefined"]
    )

    market_segment = st.selectbox(
        "Market Segment",
        [
            "Direct",
            "Corporate",
            "Online TA",
            "Offline TA/TO",
            "Complementary",
            "Groups",
            "Undefined",
            "Aviation"
        ]
    )

    distribution_channel = st.selectbox(
        "Distribution Channel",
        [
            "Direct",
            "Corporate",
            "TA/TO",
            "Undefined",
            "GDS"
        ]
    )

    is_repeated_guest = st.selectbox(
        "Repeated Guest",
        [0, 1]
    )

    previous_cancellations = st.number_input(
        "Previous Cancellations",
        min_value=0,
        value=0
    )

    previous_bookings_not_canceled = st.number_input(
        "Previous Bookings Not Cancelled",
        min_value=0,
        value=0
    )

with col3:
    reserved_room_type = st.text_input(
        "Reserved Room Type",
        "A"
    )

    assigned_room_type = st.text_input(
        "Assigned Room Type",
        "A"
    )

    booking_changes = st.number_input(
        "Booking Changes",
        min_value=0,
        value=0
    )

    deposit_type = st.selectbox(
        "Deposit Type",
        ["No Deposit", "Refundable", "Non Refund"]
    )

    days_in_waiting_list = st.number_input(
        "Days in Waiting List",
        min_value=0,
        value=0
    )

    customer_type = st.selectbox(
        "Customer Type",
        [
            "Contract",
            "Group",
            "Transient",
            "Transient-Party"
        ]
    )

    adr = st.number_input(
        "Average Daily Rate (ADR)",
        min_value=0.0,
        value=100.0
    )

    required_car_parking_spaces = st.number_input(
        "Required Car Parking Spaces",
        min_value=0,
        value=0
    )

    total_of_special_requests = st.number_input(
        "Total Special Requests",
        min_value=0,
        value=0
    )

    country = st.text_input(
    "Country",
    "PRT"
)


# Feature engineering must match the notebook.
total_guests = adults + children + babies
total_nights = (
    stays_in_weekend_nights +
    stays_in_week_nights
)
total_stay_cost = adr * total_nights


input_data = pd.DataFrame([{
    "hotel": hotel,
    "lead_time": lead_time,
    "arrival_date_year": arrival_date_year,
    "arrival_date_month": arrival_date_month,
    "arrival_date_week_number": arrival_date_week_number,
    "arrival_date_day_of_month": arrival_date_day_of_month,
    "stays_in_weekend_nights": stays_in_weekend_nights,
    "stays_in_week_nights": stays_in_week_nights,
    "adults": adults,
    "children": children,
    "babies": babies,
    "meal": meal,
    "country": country,
    "market_segment": market_segment,
    "distribution_channel": distribution_channel,
    "is_repeated_guest": is_repeated_guest,
    "previous_cancellations": previous_cancellations,
    "previous_bookings_not_canceled":
        previous_bookings_not_canceled,
    "reserved_room_type": reserved_room_type,
    "assigned_room_type": assigned_room_type,
    "booking_changes": booking_changes,
    "deposit_type": deposit_type,
    "days_in_waiting_list": days_in_waiting_list,
    "customer_type": customer_type,
    "adr": adr,
    "required_car_parking_spaces":
        required_car_parking_spaces,
    "total_of_special_requests":
        total_of_special_requests,
    "total_guests": total_guests,
    "total_nights": total_nights,
    "total_stay_cost": total_stay_cost,
}])


st.divider()

if st.button(
    "🔮 Predict Cancellation",
    type="primary",
    use_container_width=True
):
    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
    else:
        score = model.decision_function(input_data)[0]
        probability = 1 / (1 + np.exp(-score))

    if prediction == 1:
        st.error(
            "⚠️ Prediction: Booking is likely to be CANCELLED"
        )
    else:
        st.success(
            "✅ Prediction: Booking is likely to NOT be cancelled"
        )

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric(
            "Cancellation Probability",
            f"{probability * 100:.2f}%"
        )

    with col_b:
        st.metric(
            "Selected Model",
            model_name
        )

    st.subheader("Input Details")
    st.dataframe(
        input_data,
        use_container_width=True
    )