# 📦 Shipment Status Prediction

A Machine Learning project to predict the **status of shipments** (Delivered / Problematic) using shipment details such as origin, destination, carrier, shipment dates, weight, cost, and distance.  
The project covers the **end-to-end ML pipeline** from data exploration to deployment in a **Streamlit web app**.

---

## 🚀 Project Workflow

1. **Data Exploration & Cleaning**
   - Checked for missing values, duplicates, and outliers.
   - Handled null values in `Delivery_Date` and `Cost`.
   - Explored distributions, correlations, and class imbalance.

2. **Feature Engineering**
   - Extracted date-based features:
     - `Planned_Days` = Delivery_Date - Shipment_Date
     - `Ship_Day`, `Ship_Month`, `Ship_Year`
   - Dropped raw date columns after transformation.

3. **Preprocessing**
   - Applied `StandardScaler` and `np.log1p` to numerical features.
   - Encoded categorical variables with `OneHotEncoder`.
   - Combined preprocessing steps into a `ColumnTransformer`.

4. **Modeling**
   - Tested multiple ML models:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - AdaBoost
     - Gradient Boosting
     - XGBoost
   - Used **GridSearchCV** for hyperparameter tuning.

5. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score.
   - Tackled class imbalance with `class_weight='balanced'` and `scale_pos_weight` for XGBoost.
   - Visualized feature importance.

6. **Deployment**
   - Saved the best model using **Joblib**.
   - Built an interactive **Streamlit App** where users can:
     - Input shipment details.
     - Get a real-time prediction of shipment status.
   - Custom UI with a **green theme** 🌿.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy, Scikit-learn**
- **XGBoost**
- **Matplotlib, Seaborn** (for visualization)
- **Streamlit** (for deployment)
- **Joblib** (for model saving)

---

## 📂 Project Structure
├── data/ # Dataset
├── notebook for EDA & modeling
├── models/ # Saved trained models (joblib files)
├── app.py # Streamlit web app
└── README.md # Project documentatio

---

## ⚡ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/shipment-status-prediction.git
   cd shipment-status-prediction
2. Install dependencies:  pip install -r requirements.txt
3. Run the Streamlit app:   streamlit run app.py

Results
-Best Model: XGBoost / Gradient Boosting
-Accuracy: ~85%
-Improved minority class handling using class weighting.
🌐 Demo:https://drive.google.com/file/d/1ViKxeA3HrVN_Cdwo8Xa7YQOqH4-ef1cB/view?usp=sharing

👉 The app allows users to input shipment details and predict whether the shipment is Delivered ✅ or Problematic ⚠️.
Esraa Mahmoud
🎓 AI & Data Science Student @ Ain Shams University
🚀 USAID Pioneer Alumini | Machine Learning Enthusiast
