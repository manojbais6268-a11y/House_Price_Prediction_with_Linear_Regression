# 🏠 House Price Prediction with Linear Regression

> **OIBSIP – Data Analytics Track | Level 2 – Task 1**

A complete end-to-end machine learning project for predicting house prices using **Linear Regression**, with data cleaning, exploratory data analysis, feature engineering, categorical encoding, model evaluation, residual analysis, coefficient interpretation, and regularization comparison.

---

## 📌 Project Overview

The objective of this project is to build and evaluate a **Linear Regression model** that predicts house prices using property characteristics such as:

* Living area
* Number of bedrooms
* Number of bathrooms
* Lot size
* Floors
* Waterfront availability
* View
* Property condition
* Basement area
* Year built
* Renovation information
* City/location

The project follows an end-to-end workflow:

**Data Loading → EDA → Data Cleaning → Feature Engineering → Encoding → Correlation Analysis → Model Training → Evaluation → Residual Analysis → Coefficient Analysis → Regularization Comparison**

---

## 📂 Dataset

The project uses a `data.csv` dataset containing **4,600 house sales** from the **King County / Seattle area, Washington, USA**, covering **May–July 2014**.

### Dataset Dimensions

* **Rows:** 4,600
* **Columns:** 18

### Main Features

| Feature         | Description                |
| --------------- | -------------------------- |
| `date`          | Sale date                  |
| `price`         | Target house price         |
| `bedrooms`      | Number of bedrooms         |
| `bathrooms`     | Number of bathrooms        |
| `sqft_living`   | Living area in square feet |
| `sqft_lot`      | Lot area                   |
| `floors`        | Number of floors           |
| `waterfront`    | Waterfront indicator       |
| `view`          | View rating                |
| `condition`     | Property condition         |
| `sqft_above`    | Above-ground living area   |
| `sqft_basement` | Basement area              |
| `yr_built`      | Year built                 |
| `yr_renovated`  | Year renovated             |
| `street`        | Street address             |
| `city`          | City                       |
| `statezip`      | State and ZIP code         |
| `country`       | Country                    |

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook / Google Colab**

### Machine Learning Techniques

* Linear Regression
* Ridge Regression
* Lasso Regression
* One-Hot Encoding
* Standardization
* Train/Test Split
* Residual Analysis

---

# 🔎 Exploratory Data Analysis

The notebook begins with an initial inspection of the dataset, including:

* Dataset shape
* Data types
* Sample records
* Numerical feature statistics
* Target variable distribution
* Missing-value inspection
* Feature relationships

The target variable is **`price`**, representing the house sale price.

---

# 🧹 Data Cleaning & Feature Engineering

The project prepares the raw dataset for machine learning by:

* Inspecting data types
* Handling categorical variables
* Preparing numerical features
* Converting categorical location information into model-compatible features
* Removing extreme-price observations for the main modelling analysis

The notebook specifically excludes **89 extreme-price houses above approximately $1.65 million** from the modelling dataset.

This decision is documented because these extreme observations can strongly influence a linear regression model.

---

# 🎯 Feature Selection

The project considers property characteristics that can reasonably contribute to house prices.

Important predictive variables include:

* `sqft_living`
* `bedrooms`
* `bathrooms`
* `sqft_lot`
* `floors`
* `waterfront`
* `view`
* `condition`
* `sqft_above`
* `sqft_basement`
* `yr_built`
* `yr_renovated`
* `city`

Location is particularly important because house prices can vary substantially between cities.

---

# 📊 Correlation Analysis

A correlation heatmap is used to examine relationships between numerical variables and the target variable.

The analysis helps identify which numerical characteristics have stronger relationships with house prices and provides guidance for feature selection.

---

# ⚙️ Machine Learning Workflow

The modelling pipeline follows these steps:

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Train/Test Split
     ↓
Linear Regression
     ↓
Model Evaluation
     ↓
Residual Analysis
     ↓
Coefficient Analysis
     ↓
Ridge & Lasso Comparison
```

The dataset is divided into:

* **80% Training Data**
* **20% Test Data**

A fixed random state is used to make the experiment reproducible.

---

# 🤖 Linear Regression Model

The primary model is:

```python
LinearRegression()
```

The model learns the relationship between house characteristics and the target variable:

```text
House Features → Predicted House Price
```

---

# 📏 Model Evaluation

The model is evaluated using:

* **MSE – Mean Squared Error**
* **RMSE – Root Mean Squared Error**
* **R² – Coefficient of Determination**
* **MAE – Mean Absolute Error**

### Test Set Performance

| Metric   |      Result |
| -------- | ----------: |
| **R²**   | ≈ **0.705** |
| **RMSE** | ≈ **$144K** |
| **MAE**  | ≈ **$100K** |

The notebook reports that approximately **62% of predictions fall within ±20% of the actual house price**.

The training and test performance are reported as being very similar, indicating that the model does not show a substantial train/test performance gap in this experiment.

---

# 📈 Actual vs Predicted Prices

An actual-vs-predicted scatter plot is used to compare:

* Actual house prices
* Model-predicted house prices

A prediction closer to the diagonal reference line represents closer agreement between predicted and actual prices.

---

# 📉 Residual Analysis

Residual analysis is performed to examine prediction errors.

The analysis identifies:

* Increasing error for more expensive properties
* A heavy right tail in the residual distribution
* Evidence of heteroscedasticity

Therefore, the model's errors are not completely uniform across the entire price range.

---

# 📌 Coefficient Analysis

The regression coefficients are analysed to understand the direction and relative contribution of model features.

The notebook identifies:

### Major Price Drivers

* **Location / City**
* **Living area (`sqft_living`)**

### Additional Contributors

* View
* Bathrooms

The city-related coefficients are particularly important because location contributes substantially to differences in house prices.

---

# 🔬 Regularization Comparison

As a bonus analysis, the project compares:

* Linear Regression
* Ridge Regression
* Lasso Regression

### Test Performance

| Model             |       R² |      RMSE |
| ----------------- | -------: | --------: |
| Linear Regression | ≈ 0.7046 | ≈ $144.4K |
| Ridge Regression  | ≈ 0.7057 | ≈ $144.0K |
| Lasso Regression  | ≈ 0.7059 | ≈ $144.0K |

The notebook finds that the three models have **very similar predictive performance** on this dataset.

However, Ridge and Lasso affect the coefficients differently:

* **Ridge** shrinks correlated coefficients.
* **Lasso** can shrink some coefficients exactly to zero.

---

# 📊 Additional Log-Price Experiment

The notebook also evaluates a model using the logarithm of house prices.

The log-price model achieves approximately:

* **R² ≈ 0.73 on the log scale**

However, when evaluated in dollar terms, its performance is reported as approximately:

* **R² ≈ 0.66**
* **RMSE ≈ $155K**

Therefore, the notebook retains the plain-price modelling approach when the objective is dollar-level prediction accuracy.

---

# 💡 Key Findings

Based on the analysis performed in the notebook:

1. **Location and living area are major contributors to house prices.**

2. **View and bathrooms provide additional price contribution**, although their effect is smaller than the main drivers.

3. The Linear Regression model achieves approximately **0.705 R²** on the test set after removing extreme-price observations.

4. Prediction errors become larger for expensive houses, indicating limitations of the linear model across the full price range.

5. Ridge and Lasso provide very similar predictive performance to ordinary Linear Regression for this dataset.

6. The dataset does not contain geographic coordinates, neighbourhood-quality variables, or detailed interior-finish information, limiting the model's ability to capture all sources of price variation.

---

# ⚠️ Model Limitations

The notebook identifies several limitations:

* The dataset represents a relatively short **2014 sales period**.
* Geographic latitude/longitude information is unavailable.
* Neighbourhood-quality information is not included.
* Interior finish information is unavailable.
* The model is not intended for luxury properties because extreme-price observations above approximately **$1.65M** were excluded.
* Residual errors increase for expensive houses.
* The linear model may not capture complex non-linear relationships between property characteristics and price.

---

# 🚀 Future Improvements

The notebook suggests several possible extensions:

### 1. Interaction Features

Introduce interactions such as:

```text
City × Living Area
```

to capture location-dependent effects of property size.

### 2. Tree-Based Models

Experiment with:

* Random Forest
* Gradient Boosting
* Other non-linear regression techniques

### 3. Geographic Features

Add:

* Latitude
* Longitude
* Neighbourhood information

These could provide additional location-level predictive information.

### 4. Advanced Feature Engineering

Additional property-level features could be engineered to improve model representation and potentially reduce systematic prediction errors.

---

# 📁 Project Structure

```text
House-Price-Prediction/
│
├── data.csv
├── House_Price_Prediction.ipynb
└── README.md
```

> Update the notebook filename above if your GitHub repository uses a different filename.

---

# ▶️ How to Run

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the project folder

```bash
cd <PROJECT_FOLDER>
```

### 3. Install required libraries

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the notebook

Open the project `.ipynb` file and run the cells sequentially.

---

# 🎓 Learning Outcomes

This project provided practical experience in:

* Exploratory Data Analysis
* Data Cleaning
* Feature Engineering
* Categorical Encoding
* Correlation Analysis
* Linear Regression
* Model Evaluation
* RMSE, MSE, MAE and R²
* Residual Analysis
* Regression Coefficient Interpretation
* Ridge Regression
* Lasso Regression
* Model Limitations and Improvement Strategies

---

# 📌 OIBSIP Internship

**Program:** Oasis Infobyte Internship Program
**Track:** Data Analytics
**Level:** Level 2
**Task:** Task 1 – Predicting House Prices with Linear Regression

---

## 👤 Author

**Manoj Kumar Bais**

MCA (AI & ML)
Bhopal, Madhya Pradesh, India

### Connect

* **GitHub:** https://github.com/manojbais6268-a11y
* **LinkedIn:** https://www.linkedin.com/in/manoj-kumar-86a560322

---

## 📄 Project Summary

This project demonstrates an end-to-end regression workflow for house-price prediction, beginning with exploratory analysis and data preparation and progressing through Linear Regression modelling, evaluation, residual diagnostics, coefficient interpretation, and comparison with Ridge and Lasso regularization.

The final Linear Regression model achieved approximately **R² = 0.705** with **RMSE ≈ $144K** and **MAE ≈ $100K** on the reported test set after the documented removal of extreme-price observations.
