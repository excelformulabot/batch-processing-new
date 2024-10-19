import json
import os
import requests
import pandas as pd
import concurrent.futures
import time
from threading import Lock
import csv
import boto3
import io  # For using StringIO
from flask import Flask, request, jsonify
# from dotenv import load_dotenv

# # Load the environment variables from the .env file
# load_dotenv()
# OpenAI API settings
openai_api_key = os.getenv('OPENAI_API_KEY')
base_url = "https://api.openai.com/v1"
headers = {
    "Authorization": f"Bearer {openai_api_key}",
    "Content-Type": "application/json"
}
bucket_name = os.getenv('AWS_S3_BUCKET')

# S3 client setup (use your credentials or IAM role)
# s3_client = boto3.client('s3')
s3_client = boto3.client('s3', 
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY'), 
                      aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
                      region_name=os.getenv('AWS_REGION'))

# Initialize a global counter and a lock for thread safety
processed_records_counter = 0
counter_lock = Lock()

# Flask app setup
app = Flask(__name__)


def count_errors_in_csv(csv_buffer):
    csv_buffer.seek(0)  # Go back to the beginning of the buffer
    error_count = 0
    error_indexes = []
    csv_reader = csv.reader(csv_buffer)
    
    # Skip the header row
    next(csv_reader)
    
    for idx, row in enumerate(csv_reader):
        if row[-1] == "Error: Unable to process":  # The response is the last column
            error_count += 1
            error_indexes.append(idx + 1)  # Add 1 to the index to account for the header row

    return error_count, error_indexes
    
# Function to make OpenAI API call with retries
def make_openai_call(prompt, model, temperature, tokens):
    url = f"{base_url}/chat/completions"
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens,
        "temperature": temperature
    }

    retries = 7
    delay = 1  # Start with a 1-second delay between retries

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=200)
            response.raise_for_status()
            response_json = response.json()
            return response_json['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            print(f"Error during OpenAI API call: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)  # Wait before retrying
            delay *= 2  # Exponential backoff: double the delay each retry

    return "Error: Unable to process"

# Function to process a single row
def process_row(index, row, column_index, system_prompt, model, temperature, categories, tokens):
    global processed_records_counter
    try:
        input_text = str(row[column_index])
        prompt = f"{system_prompt}\n\nInput: {str(input_text)}"

        print(f"Processing record at index {index} with content: {input_text[:50]}...")
        start_time = time.time()

        response = make_openai_call(prompt, model, temperature, tokens)

        elapsed_time = time.time() - start_time
        print(f"Completed processing for index {index}. Time taken: {elapsed_time:.2f} seconds")

        # Thread-safe counter update
        with counter_lock:
            processed_records_counter += 1
            if processed_records_counter % 100 == 0:
                print(f"Processed {processed_records_counter} records so far.")

        return index, response

    except Exception as e:
        print(f"Error processing row at index {index}: {e}")
        return index, "Error: Unable to process"

# Function to process a batch of rows
def process_batch(batch, column_index, system_prompt, model, temperature, row_max_workers, categories, tokens):
    print(f"Processing batch with {len(batch)} records started...")
    start_time = time.time()

    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=row_max_workers) as executor:
            futures = {
                executor.submit(process_row, idx, row, column_index, system_prompt, model, temperature, categories, tokens): idx
                for idx, row in batch.iterrows()
            }

            for future in concurrent.futures.as_completed(futures, timeout=3600):
                try:
                    index, response = future.result(timeout=3600)
                    results.append((index, response))
                except concurrent.futures.TimeoutError:
                    print(f"TimeoutError for record at index {futures[future]}. Skipping.")
                except Exception as e:
                    print(f"Error processing record at index {futures[future]}: {e}")

    except Exception as e:
        print(f"Batch processing error: {e}")

    results.sort(key=lambda x: x[0])

    end_time = time.time()
    print(f"Completed batch processing in {end_time - start_time:.2f} seconds")
    return results

# Main function to process the request
@app.route('/process_csv', methods=['POST'])
def process_csv():
    global processed_records_counter
    processed_records_counter = 0  # Reset counter

    # Get parameters from the request JSON body
    try:
        event = request.json
        csv_url = event['s3_file_url']
        print(csv_url)
        column_index = int(event['column_index'])
        max_rows = int(event['max_rows'])
        system_prompt = event['system_prompt']
        model = event['model']
        temperature = float(event['temperature'])
        tokens = int(event['tokens'])
        file_name = event['file_name']
        request_id = event['request_id']
        version_id = event['version_id']
    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {str(e)}"}), 400

    batch_size = 1000
    batch_max_workers = 10
    row_max_workers = 20
    print(f"Starting CSV processing from URL: {csv_url}")

    # Load the CSV
    try:
        raw_data = pd.read_csv(csv_url)
        orig_data = raw_data
    except Exception as e:
        return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 500

     
    categories = []

    length_of_csv = len(raw_data)
    print(f"Original CSV has {str(length_of_csv)} records")

    # Prepare to write CSV data to buffer instead of a file
    csv_buffer = io.StringIO()

    # Write the header to the buffer
    csv_writer = csv.writer(csv_buffer)
    header = list(raw_data.columns) + ['Response']
    csv_writer.writerow(header)

    # Limit rows
    raw_data = raw_data.head(max_rows)

    # Create batches
    num_batches = (len(raw_data) + batch_size - 1) // batch_size
    batches = [raw_data[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    print(f"Processing {num_batches} batches with a batch size of {batch_size}.")

    # Start processing
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_max_workers) as batch_executor:
        all_results = []
        batch_futures = [
            batch_executor.submit(process_batch, batch, column_index, system_prompt, model, temperature, row_max_workers, categories, tokens)
            for batch in batches
        ]
        print("Batch Futures : ",batch_futures)
        for future in concurrent.futures.as_completed(batch_futures):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception as e:
                print(f"Error during batch processing: {e}")
                return jsonify({"error": f"Error during batch processing: {str(e)}"}), 500

        print("All results for loop done! ")
        # print("all results: ",all_results)

    # print("all results: ",all_results)
    # Write the results to the buffer
    print("About to sort! ")
    all_results.sort(key=lambda x: x[0]) # Sort by original index
    print("Sorted! ")
    # print("all results after sorting : ",all_results)
    for index, response in all_results:
        if index >= len(raw_data):
            print(f"Index {index} is out of bounds for raw_data with length {len(raw_data)}")
            continue  # Skip this index if it's out of bounds
        
        row_data = raw_data.iloc[index].tolist()
        row_data.append(response)
        csv_writer.writerow(row_data)

    
    remaining_rows = orig_data.iloc[max_rows:]  # Get rows beyond max_rows

    for _, remaining_row in remaining_rows.iterrows():
        remaining_row_data = remaining_row.tolist()
        remaining_row_data.append('')  # Append an empty response
        csv_writer.writerow(remaining_row_data)


    error_count, error_indexes = count_errors_in_csv(csv_buffer)
    print(f"Number of records with 'Error: Unable to process': {error_count}")
    print(f"Indexes with errors: {error_indexes}")

    print("Reached here")
    # Upload to S3 using the buffer
    try:
        file_key = file_name + "_final.csv"
        csv_buffer.seek(0)
        s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue(), ACL='private')
        csv_url = f"https://{bucket_name}.s3.amazonaws.com/{file_key}"
        print(f"Uploading file to S3 bucket: {bucket_name}, key: {file_key}")
    except Exception as e:
        return jsonify({"error": f"Error uploading to S3: {str(e)}"}), 500

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds & file uploaded to S3.")

    print("Calling the webhook url now with our response ")
    response = jsonify({"message": "Processing completed", "file_url": csv_url, "error_count": error_count, "request_id": request_id})

    webhook_url = "https://excel-formula-bot2.bubbleapps.io/version-" + version_id + "/api/1.1/wf/data-enrichment/"

    try:
        response_generated_from_webhook_url = requests.post(webhook_url, json=response, timeout=200)
        response_generated_from_webhook_url.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error during post call to webhook url: {e}")

    return response

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
