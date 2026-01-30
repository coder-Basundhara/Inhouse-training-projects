# Customer Spending Prediction Using Decision Tree Regression

## 🚀 Project Overview
This project predicts **Yearly Amount Spent** by customers of an e-commerce platform using behavioral and engagement metrics.  
It demonstrates a **complete ML pipeline** with emphasis on **interpretability**, **robustness to unseen data**, and **FAANG-level best practices**.

---

## 📝 Problem Statement
Given customer usage metrics like session behavior and membership duration, predict their **annual spending**.

- **Type:** Supervised Learning (Regression)  
- **Target:** `Yearly Amount Spent` (continuous numeric)

---

## 📊 Dataset Description

| Feature | Type | Description |
|---------|------|------------|
| Email | Categorical (ID) | Customer identifier |
| Address | Text | Residential address |
| Avatar | Categorical | Customer UI theme |
| Avg. Session Length | Numerical | Average session duration |
| Time on App | Numerical | Daily app usage |
| Time on Website | Numerical | Daily website usage |
| Length of Membership | Numerical | Customer tenure |
| Yearly Amount Spent | Numerical | Target variable |

**Sample Record:**
Avatar: Violet
Avg. Session Length: 34.49
Time on App: 12.65
Time on Website: 39.57
Length of Membership: 4.08
Yearly Amount Spent: 587.95


---

## 🧹 Data Cleaning & Feature Selection
**Removed features:**
- `Email` → Unique identifier, no predictive value
- `Address` → High cardinality, noisy

**Retained features:**
- `Avatar` (categorical, one-hot encoded)
- `Avg. Session Length`, `Time on App`, `Time on Website`, `Length of Membership` (numerical)

---

## ⚙️ Preprocessing
- One-Hot Encoding for `Avatar` with `handle_unknown="ignore"`  
- No scaling required for numerical features (Decision Trees are scale-invariant)  

---

## 🛠 Train-Test Strategy
- 80% training, 20% testing  
- `random_state=42` for reproducibility  

---

## 🔧 Model Selection
**Decision Tree Regressor** chosen for:  
- Interpretability  
- Handling nonlinear relationships  
- Native feature importance support  

**Hyperparameters:**
```text
max_depth = 5
min_samples_split = 10
min_samples_leaf = 5
random_state = 42

🏗 Model Architecture
Input Features
   ↓
One-Hot Encoder (Avatar)
   ↓
Decision Tree Regressor
   ↓
Predicted Yearly Amount Spent

📏 Evaluation Metrics
Metric	Purpose
MAE	Average absolute error
MSE	Penalizes large errors
RMSE	Error in original units
R² Score	Variance explained by model
🌈 Explainability

Decision Tree Visualization: Color-filled nodes indicate predicted spending; darker = higher.

Feature Importance:

Length of Membership

Time on App

Avg. Session Length

Feature importance helps explain which factors most influence customer spending.

🔍 Handling Unseen Data

Unknown Avatar values handled via handle_unknown="ignore"

Model generalizes safely to completely new input without errors

Example:

unknown_customer = pd.DataFrame({
    "Avatar": ["UltraBlue"],
    "Avg. Session Length": [36.4],
    "Time on App": [14.8],
    "Time on Website": [41.2],
    "Length of Membership": [5.1]
})
prediction = model.predict(unknown_customer)

⚡ Limitations

Single decision tree may overfit small datasets

Sensitivity to small variations in data

🔮 Future Enhancements
Improvement	Benefit
Random Forest	Better generalization
Gradient Boosting	Higher accuracy
SHAP values	Advanced explainability
Cross-validation	Robust evaluation
🖥 Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib

Google Colab

💡 Key Learnings

Importance of feature selection

Interpretable models are crucial for business ML

Robust handling of unseen categorical data

End-to-end ML pipeline design for production-readiness


