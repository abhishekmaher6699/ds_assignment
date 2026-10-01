# Hotel Booking Cancellation Prediction

## 1. Project Overview

This project uses machine learning to predict whether a hotel booking will be cancelled.

The project compares four classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Random Forest is further optimized using GridSearchCV.

A Streamlit application provides an interactive interface for making predictions.

---

## 2. Problem Statement

Hotel cancellations can affect room availability, revenue planning, and operational decisions.

The objective of this project is to build a machine learning classification system that predicts whether a hotel reservation will be cancelled based on information available about the booking.

---

## 3. Dataset

The project uses the **Hotel Booking Demand** dataset.

The dataset contains hotel reservation records with information about:

- Hotel type
- Lead time
- Arrival date
- Length of stay
- Number of guests
- Meal plan
- Market segment
- Distribution channel
- Previous cancellations
- Deposit type
- Customer type
- Average daily rate
- Special requests
- Room information

### Target Variable

`is_canceled`

- `0` → Booking was not cancelled
- `1` → Booking was cancelled

---

## 4. Machine Learning Workflow

The project follows this workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Preprocessing
   ↓
Train/Test Split
   ↓
Four Classification Models
   ↓
Model Evaluation
   ↓
Random Forest Hyperparameter Tuning
   ↓
Final Model
   ↓
Streamlit Application
```

---

## 5. Data Preprocessing

The preprocessing stage includes:

- Duplicate removal
- Missing-value handling
- Removal of invalid booking records
- Numerical feature imputation
- Numerical feature scaling
- Categorical feature imputation
- One-hot encoding

The preprocessing operations are implemented using Scikit-learn pipelines to keep training and prediction transformations consistent.

---

## 6. Feature Engineering

Additional features are created:

### Total Guests

```text
adults + children + babies
```

### Total Nights

```text
stays_in_weekend_nights + stays_in_week_nights
```

### Total Stay Cost

```text
adr × total_nights
```

---

## 7. Data Leakage Prevention

`reservation_status` and `reservation_status_date` are excluded from model training because they contain information directly associated with the final reservation outcome.

`agent` and `company` identifiers are also excluded from the predictive feature set.

---

## 8. Models

### Logistic Regression

Used as a linear classification baseline.

### Decision Tree

Used to model non-linear relationships using decision rules.

### Random Forest

An ensemble of decision trees used for robust non-linear classification.

### Support Vector Machine

A linear SVM implementation (`LinearSVC`) is used because it is considerably more computationally efficient for the large, one-hot-encoded dataset.

---

## 9. Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The models are compared using a final performance table and visualization.

---

## 10. Hyperparameter Tuning

Random Forest is optimized using `GridSearchCV` with 5-fold cross-validation.

The search explores:

- Number of trees
- Maximum tree depth
- Minimum samples required for splitting
- Minimum samples required at a leaf

F1-score is used as the GridSearchCV scoring metric.

---

## 11. Feature Importance

Feature importance from the tuned Random Forest is calculated to identify which transformed features contribute most strongly to the model's predictions.

---

## 12. Streamlit Application

The Streamlit application allows a user to:

1. Select a machine learning model.
2. Enter hotel booking information.
3. Submit the booking details.
4. Receive a cancellation prediction.
5. View the estimated cancellation probability.

Run the application using:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 13. Project Structure

```text
hotel-booking-ml/
│
├── hotel_bookings.csv
├── hotel_booking_cancellation.ipynb
├── app.py
├── requirements.txt
├── README.md
│
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── random_forest.pkl
    ├── svm.pkl
    └── tuned_random_forest.pkl
```

---

## 14. Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook hotel_booking_cancellation.ipynb
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 15. Conclusion

This project demonstrates an end-to-end machine learning workflow for hotel booking cancellation prediction, from data preprocessing and exploratory analysis to model comparison, hyperparameter tuning, feature analysis, and interactive deployment using Streamlit.
