#(C)Tsubasa Kato - Inspire Search Corporation 2023/6/26
#Created with the help of ChatGPT (GPT-4)
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForTokenClassification
import torch
import pandas as pd
import json
import asyncio
import aiohttp
import aiofiles
import async_timeout

# Load pre-trained BERT tokenizer and model for token classification
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to GPU if available
model = model.to(device)

# Define the labels
labels = ['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

# Function to fetch webpage and process text
async def crawl_and_process(session, url):
    word_labels = []
    #Sleep for 1 second
    await asyncio.sleep(1)
    
    try:
        print("Async fetch: " + url)
        # Specify a timeout of 10 seconds for the request
        async with async_timeout.timeout(10):
            async with session.get(url) as response:
                response.raise_for_status()
                soup = BeautifulSoup(await response.text(), 'html.parser')
                text = soup.get_text()
        # Tokenize the text and move to GPU
        tokens = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        tokens = tokens.to(device)

        # Predict all tokens
        print("Predicting tokens: " + url)
        with torch.no_grad():
            predictions = model(tokens['input_ids'])

        predicted_index = torch.argmax(predictions[0], dim=2)
        predicted_index = predicted_index.to('cpu')

    except Exception as e:
        print(f"Failed to fetch and process {url}. Error: {e}")
        return (url, word_labels)

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

    return (url, word_labels)

# Create a function to run the asyncio event loop
async def main(urls):
    # Create a session
    async with aiohttp.ClientSession() as session:
        tasks = []
        # Assign tasks
        for url in urls:
            tasks.append(crawl_and_process(session, url))
        # Run tasks and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

# Read list of URLs from text file
with open('urls.txt', 'r') as f:
    urls = [line.strip() for line in f]

# Run the asyncio event loop
results = asyncio.run(main(urls))

# Process the results and save them to a CSV file
df = pd.DataFrame(columns=['URL', 'Words'])
for res in results:
    url, word_labels = res
    df_temp = pd.DataFrame({'URL': [url], 'Words': [json.dumps(word_labels)]})
    df = pd.concat([df, df_temp], ignore_index=True)
df.to_csv('crawl_results.csv', index=False)

