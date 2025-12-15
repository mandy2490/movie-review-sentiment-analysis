import os
import re
import joblib
import pandas as pd
from io import BytesIO
from google.cloud import bigquery
from google.cloud import storage
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Configuration
PROJECT_ID = "ninth-nebula-475418-p4"
BUCKET_NAME = "movie_review_sentiment_analysis"
MODEL_PATH = "sentiment_model_artifact/logistic_regression_model.joblib"
VECTORIZER_PATH = "sentiment_model_artifact/tfidf_vectorizer.joblib"

bq_client = bigquery.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)

def clean_text(text):
    if text is None:
        return ""
        
    text = text.lower()
    text = re.sub('<br />', ' ', text)
    
    # Remove URLs, mentions, hashtags, punctuation, and numbers
    text = re.sub(r"https?\S+|www\.\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r'\d+', '', text)
    
    tokens = text.split()
    filtered = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    
    return " ".join(filtered)

def run_notebook_prediction():
    print("Starting batch prediction...")

    # Load artifacts from GCS
    bucket = storage_client.bucket(BUCKET_NAME)
    
    blob_model = bucket.blob(MODEL_PATH)
    model_file = BytesIO()
    blob_model.download_to_file(model_file)
    model = joblib.load(model_file)
    
    blob_vec = bucket.blob(VECTORIZER_PATH)
    vec_file = BytesIO()
    blob_vec.download_to_file(vec_file)
    vectorizer = joblib.load(vec_file)
    print("Model and vectorizer loaded.")

    # Fetch unscored data
    query = """
        SELECT review_id, review 
        FROM `ninth-nebula-475418-p4.movie_review.reviews_movie_info` 
        WHERE predicted_sentiment IS NULL
        LIMIT 1000
    """
    df = bq_client.query(query).to_dataframe()
    
    if df.empty:
        print("No new data to process.")
        return

    print(f"Processing {len(df)} reviews...")

    # Preprocessing and Prediction
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    X_tfidf = vectorizer.transform(df['cleaned_review'])
    df['predicted_sentiment'] = model.predict(X_tfidf)

    # Save results via temp table merge
    upload_df = df[['review_id', 'predicted_sentiment']].astype({'predicted_sentiment': int})
    temp_table_id = "ninth-nebula-475418-p4.movie_review.temp_updates"
    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    bq_client.load_table_from_dataframe(upload_df, temp_table_id, job_config=job_config).result()

    merge_sql = f"""
        MERGE `ninth-nebula-475418-p4.movie_review.reviews_movie_info` T
        USING `{temp_table_id}` S
        ON T.review_id = S.review_id
        WHEN MATCHED THEN
          UPDATE SET predicted_sentiment = S.predicted_sentiment
    """
    bq_client.query(merge_sql).result()
    
    print(f"Success: Updated {len(df)} rows.")

if __name__ == "__main__":
    run_notebook_prediction()