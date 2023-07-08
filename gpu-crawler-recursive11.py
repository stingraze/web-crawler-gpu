import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForTokenClassification
import torch
import pandas as pd
import json
import asyncio
import aiohttp
from urllib.parse import urljoin
import signal
import atexit
import async_timeout
import re

visited_urls = set()
results = []  # Define the results list
append_interval = 10  # Append to CSV every 10 sites
csv_filename = 'crawl_results.csv'  # CSV file name
fetch_timeout = 3  # Fetch timeout in seconds
concurrency_limit = 10  # Number of concurrent requests

# URL exclusion list using regular expressions
exclusion_list = [
    r'^https://en\.wikipedia\.org',
    # Add more regular expressions for URLs to exclude
]

# Create a semaphore to limit the number of concurrent requests
semaphore = asyncio.Semaphore(concurrency_limit)

# Function to fetch webpage and process text
async def crawl_and_process(session, url, depth=0):
    word_labels = []

    try:
        # Pause for 1 second
        await asyncio.sleep(1)
        print("Async fetch:" + url)

        # Check if the URL matches any regex pattern in the exclusion list
        if any(re.search(pattern, url) for pattern in exclusion_list):
            print(f"URL {url} matches exclusion pattern. Skipping.")
            return

        # Fetch URL with a timeout of 3 seconds
        with async_timeout.timeout(fetch_timeout):
            async with session.get(url) as response:
                response.raise_for_status()
                html = await response.text()

        soup = BeautifulSoup(html, 'html.parser')

        text = soup.get_text()

        # Tokenize the text and move to GPU
        tokens = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        tokens = tokens.to(device)

        # Predict all tokens with a timeout of 10 seconds
        signal.alarm(10)
        print("Predicting tokens for " + url)
        with torch.no_grad():
            predictions = model(tokens['input_ids'])

        signal.alarm(0)  # Reset the alarm

        predicted_index = torch.argmax(predictions[0], dim=2)
        predicted_index = predicted_index.to('cpu')

        current_word, current_label = "", "O"
        for token, prediction in zip(tokens['input_ids'][0], predicted_index[0]):
            decoded_token = tokenizer.decode([token.item()]).strip()

            if decoded_token.startswith("##"):
                # This token is a subtoken of a larger word
                current_word += decoded_token[2:]
            else:
                # This token is a new word; save the old word (if it's not an 'O' entity)
                if current_label != 'O':
                    word_labels.append({current_word: current_label})
                current_word = decoded_token
                current_label = labels[prediction]

        # Save the last word (if it's not an 'O' entity)
        if current_label != 'O':
            word_labels.append({current_word: current_label})

        results.append((url, word_labels))  # Append results

        # Extract all links if depth is less than or equal to 1
        if depth <= 1:
            for link in soup.find_all('a'):
                new_url = link.get('href')
                if new_url:
                    new_url = urljoin(url, new_url)
                    if new_url not in visited_urls:
                        visited_urls.add(new_url)
                        await crawl_and_process(session, new_url, depth + 1)  # Recursive call

    except asyncio.TimeoutError:
        print(f"Timeout occurred for {url}. Moving on to the next URL.")

    except Exception as e:
        print(f"Failed to fetch and process {url}. Error: {e}")

# Create a function to run the asyncio event loop
async def main(seed_urls):
    # Create a session
    async with aiohttp.ClientSession() as session:
        # Assign tasks with limited concurrency
        tasks = []
        for url in seed_urls:
            async with semaphore:
                task = asyncio.create_task(crawl_and_process(session, url))
                tasks.append(task)

        await asyncio.gather(*tasks)  # Await all tasks

# Load pre-trained BERT tokenizer and model for token classification
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to GPU if available
model = model.to(device)

# Define the labels
labels = ['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

# Read list of URLs from text file
with open('seeds.txt', 'r') as f:
    seed_urls = [line.strip() for line in f]

# Register the cleanup function to handle saving the data
@atexit.register
def save_data_on_exit():
    # Process the results and save them to a CSV file
    df = pd.DataFrame(columns=['URL', 'Words'])
    for res in results:
        url, word_labels = res
        df_temp = pd.DataFrame({'URL': [url], 'Words': [json.dumps(word_labels)]})
        df = pd.concat([df, df_temp], ignore_index=True)

    # Save the data to the CSV file
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

# Run the asyncio event loop
asyncio.run(main(seed_urls))
