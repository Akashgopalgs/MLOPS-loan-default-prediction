# 🏦 MLOps Loan Default Prediction

An end-to-end, production-grade machine learning project that predicts loan default risk — from raw data ingestion to live deployment via a FastAPI application hosted on AWS.

---

## 🚀 Project Objective

To demonstrate a real-world MLOps pipeline that handles class imbalance, builds robust machine learning models, integrates CI/CD workflows, and deploys a live prediction system using FastAPI and AWS services.

---
## 👨‍💻 Tech Stack

| Area                        | Skills & Tools Used                                                  |
|----------------------------|-----------------------------------------------------------------------|
| Data Preprocessing         | Python, Pandas, SMOTE, Sklearn                                       |
| Model Development          | Logistic Regression, Random Forest, XGBoost                          |
| Backend API Development    | FastAPI, Uvicorn, Joblib                                             |
| MLOps & Deployment         | Docker, GitHub Actions (CI/CD), AWS EC2, AWS S3, Bash                |
| UI Integration             | HTML (Jinja2), CSS, Basic form handling                              |
| System Design & Automation | CI/CD pipeline from GitHub to EC2, Docker image builds, error handling|

---
## 📊 Project Workflow

1. ✅ Data Preprocessing
   - Handled missing values, encoded categorical features, normalized data
   - Used **SMOTE** to manage class imbalance (~8% positive class)

2. ✅ Model Training & Evaluation
   - Trained and tuned Logistic Regression, Random Forest, and XGBoost models
   - Evaluated performance using precision, recall, and confusion matrix

3. ✅ API Development
   - Created a prediction API with **FastAPI**
   - UI built with basic HTML templates for input and display

4. ✅ Dockerization
   - Created a production-ready Docker container for the app

5. ✅ CI/CD Pipeline
   - GitHub Actions to automate testing, building, and deployment
   - Automatically uploads builds to AWS EC2 via S3

6. ✅ AWS Deployment
   - Docker containers deployed to EC2 with persistent storage via S3

---
## 🧠 Key Learnings

- Implemented robust MLOps practices using real-world cloud and automation tools
- Gained hands-on experience with deploying ML models as APIs
- Learned how to handle class imbalance and pipeline reproducibility
- Automated the build and deployment process using GitHub Actions and AWS

---
- 🌍 Live App: The live demo is currently unavailable due to AWS EC2 memory limitations.
- dataset : https://www.kaggle.com/competitions/home-credit-default-risk/data
- 💼 LinkedIn: [https://linkedin.com/in/akashgopalgs](https://linkedin.com/in/akashgopalgs)
