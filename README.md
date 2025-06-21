# **Estimating Stock Keeping Units (SKUs) Using Machine Learning**

**Overview**  
This project focuses on forecasting SKU-level demand using machine learning. By analyzing historical sales data from Kaggle, the system predicts upcoming demand for individual products to help businesses avoid stockouts, reduce overstock, and streamline inventory operations.

**Dataset**  
The dataset used in this project was sourced from Kaggle:
https://www.kaggle.com/datasets/aswathrao/demand-forecasting

**Features**
- Preprocessing handled using the EasyPreProcessing library.
- Lag features engineered to capture previous weekly sales.
- Multiple regression models evaluated: Linear Regression, Decision Tree, Random Forest, XGBoost.
- Hyperparameter tuning performed using RandomizedSearchCV.
- Final Random Forest model saved using pickle.

**Installation**  
Clone the repository:
```
git clone https://github.com/xoBerlierxo/Estimation-Stock-Prediction.git
```

**Install required packages:**  
```
pip install -r requirements.txt
```

**How to Run**  
Run the training and evaluation script from the root directory:
```
python model_training.py
```

**Results**
The tuned Random Forest model outperformed other models with the highest R² score and lowest prediction error. Visual comparisons were plotted between actual and predicted demand.

**Conclusion**
This project provides a practical, modular framework for demand forecasting at the SKU level. It is adaptable to different retail datasets and can be extended to production-ready pipelines.
