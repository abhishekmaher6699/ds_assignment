# Hotel Booking Cancellation Prediction

An end-to-end machine learning project that predicts whether a hotel reservation will be cancelled using the Hotel Booking Demand dataset.

## Live Demo

**Streamlit App:** https://hotel-cancellation-app.streamlit.app/

**GitHub Repository:** https://github.com/abhishekmaher6699/ds_assignment

## Project Overview

Hotel cancellations can affect room availability, revenue planning, and operational decisions.

This project builds a binary classification system to predict:

- `0` → Booking was not cancelled
- `1` → Booking was cancelled

Four classification models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (`LinearSVC`)

Random Forest was additionally optimized using **GridSearchCV with 5-fold cross-validation**.

The trained models are integrated into an interactive Streamlit application.

## Dataset

The project uses the **Hotel Booking Demand** dataset, containing reservation information such as hotel type, lead time, arrival date, length of stay, guests, meal plan, market segment, distribution channel, previous cancellations, deposit type, customer type, ADR, and special requests.

### Target Variable

| Value | Meaning |
|---|---|
| `0` | Booking was not cancelled |
| `1` | Booking was cancelled |

## Machine Learning Workflow

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
Model Training
   ↓
Model Evaluation
   ↓
Random Forest Hyperparameter Tuning
   ↓
Feature Importance
   ↓
Model Serialization
   ↓
Streamlit Deployment
```

## Data Preprocessing

The preprocessing stage includes:

- Duplicate removal
- Missing-value handling
- Invalid-record handling
- Numerical feature imputation
- Numerical feature scaling
- Categorical feature imputation
- One-hot encoding

Scikit-learn pipelines keep training and prediction transformations consistent.

### Data Leakage Prevention

The following fields are excluded from model training:

- `reservation_status`
- `reservation_status_date`
- `agent`
- `company`

The first two contain information associated with the final reservation outcome.

## Feature Engineering

### Total Guests

```text
total_guests = adults + children + babies
```

### Total Nights

```text
total_nights = stays_in_weekend_nights + stays_in_week_nights
```

### Total Stay Cost

```text
total_stay_cost = adr × total_nights
```

## Models

### Logistic Regression
Linear classification baseline.

### Decision Tree
Non-linear classification using decision rules.

### Random Forest
Ensemble of decision trees for non-linear classification.

### Support Vector Machine
A `LinearSVC` implementation is used for computational efficiency with the large one-hot-encoded dataset.

### Tuned Random Forest
Random Forest hyperparameters were optimized using GridSearchCV with 5-fold cross-validation and F1 scoring.

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7955 | 0.6806 | 0.4831 | 0.5651 |
| Decision Tree | 0.7929 | 0.6223 | 0.6283 | 0.6253 |
| Random Forest | 0.8393 | 0.7728 | 0.5890 | 0.6685 |
| SVM | 0.7937 | 0.6868 | 0.4590 | 0.5503 |
| **Tuned Random Forest** | **0.8401** | **0.7708** | **0.5959** | **0.6722** |

### Tuned Random Forest

- **Accuracy:** 84.01%
- **Precision:** 77.08%
- **Recall:** 59.59%
- **F1 Score:** 67.22%

Compared with the original Random Forest:

- Accuracy: `0.8393 → 0.8401`
- F1 Score: `0.6685 → 0.6722`

## Feature Importance

Highest reported feature importances:

| Feature | Importance |
|---|---:|
| `lead_time` | 0.110873 |
| `total_stay_cost` | 0.072963 |
| `adr` | 0.072675 |
| `arrival_date_day_of_month` | 0.058408 |
| `total_of_special_requests` | 0.054437 |
| `arrival_date_week_number` | 0.052803 |
| `country_PRT` | 0.040808 |
| `total_nights` | 0.033432 |
| `stays_in_week_nights` | 0.031792 |
| `market_segment_Online TA` | 0.025404 |

These are model-derived importance values and should not be interpreted as causal relationships.

## Streamlit Application

The deployed application allows users to:

1. Select a machine learning model.
2. Enter hotel booking information.
3. Submit the booking details.
4. Receive a cancellation prediction.
5. View an estimated cancellation probability.

**Live Application:** https://hotel-cancellation-app.streamlit.app/

## Project Structure

```text
ds_assignment/
├── hotel_bookings.csv
├── hotel_booking_cancellation.ipynb
├── app.py
├── requirements.txt
├── README.md
├── report.pdf
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── random_forest.pkl
    ├── svm.pkl
    └── tuned_random_forest.pkl
```

The Random Forest model files are stored using joblib compression to reduce their size for deployment.

## Installation

```bash
git clone https://github.com/abhishekmaher6699/ds_assignment.git
cd ds_assignment
pip install -r requirements.txt
streamlit run app.py
```

The local application will normally be available at:

```text
http://localhost:8501
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook
- Git & GitHub

## Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

EDA included cancellation analysis, correlation analysis, and feature-importance analysis.

## Conclusion

This project demonstrates a complete machine learning workflow for hotel booking cancellation prediction, covering data preparation, exploratory analysis, feature engineering, model training, evaluation, hyperparameter tuning, feature analysis, model serialization, and interactive deployment using Streamlit.

## Project Links

- **Live Demo:** https://hotel-cancellation-app.streamlit.app/
- **GitHub:** https://github.com/abhishekmaher6699/ds_assignment
