#(C)Tsubasa Kato - Inspire Search Corporation 2023 7/17/2023 17:31PM JST
#Created with the help of Google Bard and ChatGPT (GPT-4)
#This crawls the web and gets the BERT embeddings of the contents of each site after quantum entanglement and measurement. It will also decide how to follow link after measurement.
#Still experimental. Needs more thorough debugging.
import torch
import time
from bs4 import BeautifulSoup
from transformers import BertModel, BertTokenizer
from urllib.parse import urljoin
from qiskit import QuantumCircuit, transpile, assemble, Aer, execute
from random import randint

# Load the BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create a CUDA device, if not available use CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the model to the CUDA device
model = model.to(device)

# Initialize set of fetched URLs
fetched_urls = set()

# Function to create a quantum circuit and return the results
def run_quantum_circuit(num_shots):
    # create a quantum circuit with 2 qubits
    qc = QuantumCircuit(2, 2)  # Added 2 classical bits for measurement

    # apply Hadamard gate on the first qubit to create superposition
    qc.h(0)

    # apply CNOT gate 
    qc.cx(0, 1)

    # Measure the qubits
    qc.measure([0,1], [0,1])  # Added this line to measure the qubits

    # simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=num_shots)

    # get the result
    result = job.result()
    counts = result.get_counts(qc)

    return counts

def fetch_url(url, depth):
    global fetched_urls
    
    # If we've already fetched this URL, return immediately
    if url in fetched_urls or depth <= 0:
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

        # Recurse on the links using quantum decision
        for i, link in enumerate(links):
            num_shots = 500  # Set the number of shots here
            counts = run_quantum_circuit(num_shots)
            if counts.get('00', 0) > counts.get('11', 0):
                print("Quantum Decision: Following Link")
                fetch_url(link, depth + 1)
            else:
                print("Quantum Decision: Following Link based on Random Number.")
                random_url = randint(0, len(links) - 1)
                print("Recursive Link Follow ")
                fetch_url(links[random_url], depth + 2)
    except Exception as e:
        print(f"Failed to fetch URL: {url}")
        print(f"Error: {str(e)}")

# Start crawling at some initial URL and depth
fetch_url('https://www.tsubasakato.com', 2)
