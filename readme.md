Movie Review Sentiment Analysis on GCP

An end-to-end Machine Learning pipeline that classifies movie reviews as Positive or Negative. This project demonstrates a hybrid workflow using **Jupyter Notebooks** for training and a **Python script** for batch predictions in production, fully integrated with **Google Cloud Platform (BigQuery & Cloud Storage)**.

## Project Overview

This system reads raw movie reviews from Google BigQuery, processes the text using NLP techniques, and predicts sentiment using a Logistic Regression model.

* **Training:** Performed in a local Jupyter Notebook (`training_pipeline.ipynb`).
* **Model Storage:** Artifacts are versioned and stored in **Google Cloud Storage (GCS)**.
* **Inference:** A standalone script (`main.py`) fetches new data from BigQuery, predicts sentiment, and writes results back to the database.

---

## Repository Structure

├── training_pipeline.ipynb   # 1. Loads data, trains model, saves .joblib files, uploads to GCS.
├── main.py                   # 2. Production script. Downloads model, fetches BQ data, predicts & updates.
├── requirements.txt          #    Python dependencies.
└── README.md                 #    Project documentation.
└── sentoment_model_assests   # contains the model artifacts  



Architecture & Workflow
1. Model Training (training_pipeline.ipynb)
The notebook handles the "Data Science" portion of the lifecycle:

Ingestion: Fetches labeled training data from BigQuery.

Preprocessing: Cleans text (removes HTML, URLs, punctuation, stop words).

Vectorization: Converts text to numbers using TfidfVectorizer (10k features).

Modeling: Trains a Logistic Regression classifier (Accuracy: ~86%).

Deployment: Serializes the model and vectorizer (.joblib) and uploads them to a GCS Bucket.

2. Batch Prediction (main.py)
The script handles the "Data Engineering" portion:

Setup: Downloads the latest model.joblib and vectorizer.joblib from GCS.

Extraction: Queries BigQuery for rows where predicted_sentiment IS NULL.

Processing: Applies the exact same text cleaning logic used in training.

Prediction: Generates predictions (0 or 1).

Load: Updates the BigQuery table using a MERGE statement to fill in the missing predictions.


Future Improvements:-

Add Airflow or Cloud Scheduler to run main.py automatically every night.

Implement MLflow for better experiment tracking.

Upgrade to a Transformer model (BERT/RoBERTa) for higher accuracy.