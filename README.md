# Car-Price-Prediction
🚗 Car Price Prediction – Machine Learning Project

This project aims to predict car prices using Machine Learning techniques.
The dataset used is CarPrice_Assignment.csv, containing details about 205 cars with 26 different attributes such as car brand, fuel type, horsepower, engine size, and more.

The notebook includes:

Data loading & cleaning

Exploratory Data Analysis (EDA)

Feature engineering

Data preprocessing

Model training

Model evaluation

📂 Project Structure
📁 Car-Price-Prediction
│── 📄 CarPrice_Assignment.csv
│── 📄 notebook.ipynb
│── 📄 README.md


🔧 Technologies Used

Python

NumPy

Pandas

Matplotlib / Seaborn

Scikit-learn

Google Colab / Jupyter Notebook

📊 Objectives

Understand factors affecting car prices

Explore dataset patterns

Build a regression model to predict car prices

Evaluate model performance using:

MAE

RMSE

🧠 Key Steps in the Notebook
1️⃣ Load Data
import pandas as pd
df = pd.read_csv("CarPrice_Assignment.csv")
df.head()

2️⃣ Clean Dataset

Remove duplicates

Handle missing values

Convert categorical values

3️⃣ Visualize Dataset

Distribution plots

Correlation heatmap

Brand-wise price comparison

4️⃣ Split the Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

5️⃣ Train Model

Example:

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

6️⃣ Evaluate Model
from sklearn.metrics import r2_score
r2_score(y_test, predictions)

📈 Results

The final model is able to predict car prices based on engine features, brand, body style, and technical specifications.

You can improve the model using:

Feature scaling

Polynomial regression

Random Forest / XGBoost

▶️ How to Run the Project
Option 1: Run in Google Colab

Upload dataset

Upload notebook

Run all cells

Option 2: Run Locally
pip install -r requirements.txt
jupyter notebook

🙌 Author

Anupam Maji
Machine Learning Learner
