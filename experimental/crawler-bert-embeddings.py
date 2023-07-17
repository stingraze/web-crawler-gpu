#(C)Tsubasa Kato - Inspire Search Corporation 2023 7/17/2023 11:16AM JST
#Created with the help of Google Bard and ChatGPT (GPT-4)
#This crawls the web and gets the BERT embeddings of the contents of each site. Still experimental.
import urllib.request
import torch
import time
from bs4 import BeautifulSoup
from transformers import BertModel, BertTokenizer
from urllib.parse import urljoin

# Load the BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create a CUDA device, if not available use CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the model to the CUDA device
model = model.to(device)

# Initialize set of fetched URLs
fetched_urls = set()

def fetch_url(url, depth):
    global fetched_urls
    
    # If we've already fetched this URL, return immediately
    if url in fetched_urls or depth < 0:
        return
    fetched_urls.add(url)

    try:
        # Sleep for 1 second before processing each link
        time.sleep(1)
        
        # Get the HTML content of the website
        response = urllib.request.urlopen(url)
        html = response.read().decode('utf-8')
    
        # Use BeautifulSoup to extract the text content and links from the HTML
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        links = [urljoin(url, link.get('href')) for link in soup.find_all('a')]

        # Tokenize the text and convert to input tensors
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)

        # Compute BERT embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the embeddings of the [CLS] token
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Print the BERT embeddings
        print(f"URL: {url}\nEmbedding: {embeddings}\n")

        # Recurse on the links
        for link in links:
            fetch_url(link, depth - 1)
    except:
        print(f"Failed to fetch URL: {url}")

# Start crawling at some initial URL and depth
fetch_url('https://www.example.com', 2)
