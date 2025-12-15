import time
import warnings
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.cloud import bigquery
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError

PROJECT_ID = "ninth-nebula-475418-p4"
LOCATION = "us-central1"
DATASET_ID = "movie_review"
TABLE_ID = "reviews_movie_info"
BATCH_SIZE = 50 

warnings.filterwarnings("ignore")

bq_client = bigquery.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-flash")

def get_pending_reviews(limit):
    query = f"""
        SELECT review_id, review 
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        WHERE llm_sentiment IS NULL
        LIMIT {limit}
    """
    return bq_client.query(query).to_dataframe()

def predict_sentiment_robust(text, max_retries=3):
    prompt = f"""
    Analyze the sentiment of this movie review. 
    Classify it as exactly one of the following:
    - Strong Positive
    - Positive
    - Neutral
    - Negative
    - Strong Negative
    
    Review: "{text}"
    
    Output ONLY the category.
    """
    
    config = GenerationConfig(temperature=0.0, max_output_tokens=128)

    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt, generation_config=config)
            return response.text.strip()

        except (ResourceExhausted, ServiceUnavailable, InternalServerError):
            print(f"API busy. Pausing for 10s...")
            time.sleep(10)

        except Exception as e:
            print(f"Unknown error: {e}")
            return None

    return None

def commit_to_bigquery(df_results):
    if df_results.empty: return

    temp_table_id = f"{PROJECT_ID}.{DATASET_ID}.temp_sentiment_update"
    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    bq_client.load_table_from_dataframe(df_results, temp_table_id, job_config=job_config).result()
    
    merge_sql = f"""
        MERGE `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` T
        USING `{temp_table_id}` S
        ON T.review_id = S.review_id
        WHEN MATCHED THEN
          UPDATE SET llm_sentiment = S.llm_sentiment
    """
    bq_client.query(merge_sql).result()

if __name__ == "__main__":
    print("Starting Sentiment Analysis job...")
    total_processed = 0
    
    while True:
        df_batch = get_pending_reviews(BATCH_SIZE)
        
        if df_batch.empty:
            print("All reviews processed.")
            break
            
        print(f"Processing batch of {len(df_batch)} reviews...")
        
        results = []
        for _, row in df_batch.iterrows():
            sentiment = predict_sentiment_robust(row['review'])
            
            if sentiment:
                results.append({
                    "review_id": row['review_id'],
                    "llm_sentiment": sentiment
                })
                print(f"ID {row['review_id']}: {sentiment}")
            
            time.sleep(4.0)
            
        if results:
            commit_to_bigquery(pd.DataFrame(results))
            total_processed += len(results)
            print(f"Batch saved. Total complete: {total_processed}")