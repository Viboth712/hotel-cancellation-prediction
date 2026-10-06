# Hotel Booking Cancellation Prediction (Python, Machine Learning)

A hotel group is losing revenue because guests cancel bookings, often shortly before arrival. I built and compared two classification models that flag high-risk bookings early, so staff have time to respond, for example with reminders, deposit policies or managed overbooking.

**Tools:** Python (pandas, NumPy, scikit-learn, Matplotlib) · Jupyter
**Data:** 30,000 hotel bookings, 30 variables. 28.8% of bookings were cancelled.

---

## Key findings

- **Non-refundable deposits are the strongest warning sign.** 98% of these bookings were cancelled, compared with 27% for no-deposit bookings. That points to a specific booking channel rather than normal guest behaviour.
- **The further ahead the booking, the higher the risk.** Cancellation rises from 9% for bookings made within a week of arrival to 44% for bookings made 6 to 12 months ahead.
- **Group and online travel agent bookings cancel most** (39% and 35%). Corporate and direct bookings cancel least (13% and 15%).
- **Both models perform almost the same.** Each predicts about 80% of booking outcomes correctly (ROC-AUC 0.85). When a model flags a booking as likely to cancel, it is right about 7 times in 10.

![Cancellation by deposit type](images/01_cancellation_by_deposit_type.png)
![Cancellation by lead time](images/02_cancellation_by_lead_time.png)
![Cancellation by market segment](images/03_cancellation_by_market_segment.png)

---

## Approach

**1. Data audit.** Chose `is_canceled` as the target. Found 217 duplicate rows, 54 bookings with zero guests, extreme room rates (up to $3,992 a night) and "Undefined" values used as a second form of missing data.

**2. Cleaning and feature engineering**
- Removed duplicates and zero-guest bookings.
- Split the data into training and test sets (80/20, stratified) **before** filling gaps or scaling, so no information from the test set leaked into training.
- Capped extreme room rates and lead times at the 99th percentile of the training data.
- Created new features such as total nights, total guests, family bookings and prior cancellation rate.
- Dropped `assigned_room_type`, which is only known at check-in and would leak future information.
- One-hot encoded 9 categorical variables and scaled numeric features (89 features in total).

**3. Models**

| Metric | Logistic Regression | Decision Tree (depth 6) |
|---|---|---|
| Accuracy | 0.798 | 0.796 |
| Precision (cancelled) | 0.697 | 0.688 |
| Recall (cancelled) | 0.517 | 0.523 |
| F1 (cancelled) | 0.594 | 0.594 |
| ROC-AUC | 0.846 | 0.843 |

Five-fold cross-validation gave stable results (standard deviation 0.004 to 0.006), so the scores are not a one-off.

![ROC curve comparison](images/04_roc_curve_comparison.png)
![Decision tree top drivers](images/06_decision_tree_top_drivers.png)

---

## Limitations and next steps

- **The models miss about half of actual cancellations** (recall 0.52), because neither is weighted towards the minority "cancelled" class. Next step: use `class_weight='balanced'` or lower the decision threshold, depending on how costly a missed cancellation is compared with a false alarm.
- No hyperparameter tuning yet. A grid search over tree depth and regularisation strength is the obvious next step.

![Logistic regression confusion matrix](images/05_logistic_regression_confusion_matrix.png)

---

## Repository structure

```
├── notebook/
│   └── hotel_cancellation_prediction.ipynb   # Full analysis: audit, EDA, cleaning, models, evaluation
├── images/                                   # Key charts
└── README.md
```

The dataset was provided for coursework and is not included.

---

*Nouviboth Ra · [LinkedIn](https://www.linkedin.com/in/nouviboth-ra-792439362)*
