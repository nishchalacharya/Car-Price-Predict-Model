# 🚗 Car Selling Price Prediction 

This machine learning project aims to predict the **selling price of used cars** using supervised regression models. It was done  with the goal of building a smart and accurate system from raw data to a working predictive model.

---

## 📁 Dataset

The dataset used (`car.csv`) contains information about various used cars, including:

- Car name/model
- Year of manufacturing
- Kilometers driven
- Fuel type
- Transmission type
- Selling price (target variable) and so on..look to the notebook

---

## 🧠 Project Objectives

- Clean and prepare the dataset for machine learning.
- Use **Large Language Models (LLMs)** to assist with understanding, cleaning, and transforming the data.
- Perform **Exploratory Data Analysis (EDA)** to uncover insights and visualize key trends.
- Build and fine-tune **multiple machine learning models** to accurately predict car prices.
- Evaluate models using proper metrics and compare them with a baseline (Dummy Regressor).
- A FastAPI backend for real-time predictions
- A REST API endpoint to serve predictions

---

## 📊 Exploratory Data Analysis (EDA)

- Checked for outliers and removed extreme values to improve model robustness.
- Visualized distributions using box plots and histograms.
- Analyzed correlations between features and the target price.
- Used feature importance plots to understand the impact of each input variable.

---

## 🛠️ Feature Engineering

- Converted categorical features (like fuel type, transmission) using one-hot encoding.
- Removed or transformed less relevant features.
- Visualized and ranked features using model-based importance scores.

---

## 🤖 Models Used

| Model                   | RMSE (↓)   | MAE (↓)    | R² Score (↑) |
|------------------------|------------|------------|--------------|
| Random Forest (tuned)  | ₹65,871.70 | ₹46,050.00 | 0.9168       |
| XGBoost (tuned)        | ✅ **₹63,106.68** | ✅ **₹45,442.35** | ✅ **0.9236**       |
| Dummy Regressor (Mean) | ₹228,365.65 | ₹191,795.67 | -0.00037     |

✅ Final selected model: **XGBoost Regressor (fine-tuned)**

---

## 🧪 Evaluation Metrics

- **RMSE (Root Mean Squared Error):** Measures average prediction error — lower is better.
- **MAE (Mean Absolute Error):** Measures average absolute deviation from actual price.
- **R² Score:** Proportion of variance explained by the model — closer to 1 is ideal.

---

## 📉 Visualizations

- 📈 Actual vs Predicted scatter plot for model fit
- 📊 Feature importance bar chart from Random Forest & XGBoost
- 📦 Box plots to visualize distribution and outliers

---

## 🧠 Baseline Comparison

A `DummyRegressor` that predicts the **mean selling price** was used as a baseline.  
Its poor performance (R² ≈ 0) confirmed that the ML models learned meaningful patterns.

---

## 🧰 Tech Stack

- Python 🐍
- Pandas, NumPy, Scikit-Learn
- XGBoost, Matplotlib, Seaborn
- ChatGPT (LLM) – for guided data understanding, EDA suggestions, model insights

---

## 📌 Key Takeaways

- Outlier removal significantly improved model accuracy (RMSE dropped by over 50%)
- XGBoost outperformed Random Forest slightly after tuning
- Feature importance and visual EDA were essential for insights and model tuning
- LLMs were effective in accelerating decisions and explanations during the ML pipeline
- A FastAPI backend for real-time predictions
- A REST API endpoint to serve predictions

---


## 🚀 How to Run Locally

1. **Clone the repo**
```bash
git clone https://github.com/<your-username>/car-price-predictor.git
cd car-price-predicto
---

2. ** Install requirements **
```bash
pip install -r requirements.txt
---
3. **Run FAST API server**
```bash
uvicorn main:app --reload
---
