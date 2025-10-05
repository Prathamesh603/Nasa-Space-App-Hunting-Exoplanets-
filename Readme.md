# NASA Exoplanets Detection Project

## Overview
This repository contains the full code, datasets, and notebooks for the **NASA Space Apps Challenge 2025** project on exoplanet detection. The goal of the project is to predict and classify exoplanets using the data provided by NASA, leveraging machine learning models to enable accurate predictions and insights.

## Challenge
The challenge aimed to develop innovative solutions that utilize NASA's exoplanet datasets to identify, classify, and analyze potential exoplanets. The objective is to create a model and workflow that helps in understanding planetary characteristics and contributes to scientific discovery in the field of astronomy.

## Solution
To address this challenge, we developed a **multi-model exoplanet classification system** with the following key components:

- **Data Preprocessing**: Cleaned, normalized, and imputed missing values in the NASA dataset for optimal model training.
- **Feature Engineering**: Selected and engineered features most relevant for planetary classification.
- **Machine Learning Models**: Implemented multiple models, including:
  - Random Forest
  - XGBoost
  - Support Vector Machine
  - Logistic Regression
- **Hyperparameter Tuning**: Performed hyperparameter tuning on different models to improve performance and accuracy.
- **Evaluation**: Used metrics such as classification reports, confusion matrices, ROC curves, and precision-recall curves to evaluate models.

## Folder Structure
- `notebooks/` – Jupyter notebooks for data preprocessing, model training, evaluation, and visualization.
- `models/` – Trained model files (`.pkl`) for different datasets.
- `data/` – Raw and processed datasets including example CSVs for testing predictions.
- `web/` – Frontend pages (used only for local testing; actual deployment is separate).
- `server.py` – Backend API code for prediction and serving models.
- `requirements.txt` – Python dependencies for backend.
- `README.md` – Project documentation (this file).

## Deployment
For real-time predictions and interactive use, the project has been deployed as **separate frontend and backend services**:

1. **Frontend Repository**: [NASA Frontend Repo](https://github.com/Prathamesh603/Nasa-Frontend-)  
   - Deployed version: [Live Frontend](https://nasa-frontend-qisq.onrender.com/index.html)

2. **Backend Repository**: [NASA Backend Repo](https://github.com/Prathamesh603/Nasa-Space-App-Backend)  
   - Deployed as an API on Render to serve predictions for the frontend.

3. **Link Of The Working Web Site**: [Website Link](https://nasa-frontend-qisq.onrender.com/index.html)

### How It Works
- The frontend allows users to upload CSV files with exoplanet features.
- The backend receives the data, runs predictions using the trained models, and returns results to the frontend.
- Users can view model predictions, scatter plots, and other visualizations directly in the web interface.

## Tools and Technologies
- **Languages**: Python, JavaScript, HTML, CSS
- **Backend Framework**: FastAPI
- **Frontend Framework**: React + Vite
- **Data Handling**: Pandas, NumPy
- **Machine Learning**: scikit-learn, XGBoost
- **Visualization**: Matplotlib, Plotly
- **Deployment**: Vercel (Frontend), Render (Backend)
- **Database**: Optional CSV-based storage for prediction tracking

## AI/ML Usage
- Preprocessing and model training involved standard machine learning workflows.
- Hyperparameter tuning was applied to improve model performance.
- Multiple models were compared to select the best performing solution.

## Contribution
This repository serves as the central hub for all code, notebooks, and supporting files. For live application usage, please refer to the **frontend** and **backend** repositories listed above.

---

This project demonstrates a full-stack approach to deploying AI/ML solutions, from data analysis and model training to frontend deployment for interactive user experiences. The separation of frontend and backend ensures scalability and ease of maintenance while allowing users to explore exoplanet predictions through an intuitive web interface.

