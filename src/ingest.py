# getting the data from online source

import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Tanvi Thakre tanvi.thakre0205@gmail.com"
    }

cik = "0000320193"  # Apple
url = f"https://data.sec.gov/submissions/CIK{cik}.json"

response = requests.get(url, headers=headers)
data = response.json()

def find_filing_data(data, form_type="10-K"):
    recent=data['filings']['recent']
    for i, form in enumerate(recent['form']):
        if form==form_type:
            return {
                "accession_number": recent["accessionNumber"][i],
                "primary_document": recent['primaryDocument'][i],
                'filing_date': recent['filingDate'][i],
            }
    return None

filing=find_filing_data(data)
# print(filing)

def build_filing_url(cik, filing):
    accession_no_dashes = filing["accession_number"].replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/{filing['primary_document']}"

doc_url = build_filing_url(cik, filing)
doc_response = requests.get(doc_url, headers=headers)
# print(doc_response.status_code)

# BeautifulSoup to parse out the text from html


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose() 
    text = soup.get_text(separator=" ")
    return " ".join(text.split())  

clean_text = extract_text_from_html(doc_response.text)
# print(clean_text[:500])

import os

def save_filing_text(text: str, ticker: str, form_type: str, filing_date: str, out_dir="data/raw_filings"):
    os.makedirs(out_dir, exist_ok=True)
    year = filing_date.split("-")[0]
    filename = f"{ticker}_{form_type.replace('-', '')}_{year}.txt"
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

saved_path = save_filing_text(clean_text, "AAPL", "10-K", filing["filing_date"])
print(f"Saved to {saved_path}") ## the txt file is saved in data/raw_filings folder


from dataclasses import dataclass, asdict
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class Chunk:
    text: str
    source_id: str 
    section: str      
    chunk_id: int


def chunk_document(text: str, source_id: str, section: str = "unknown",
                    chunk_size: int = None, overlap: int = None) -> list[Chunk]:
    
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    words = text.split()
    chunks = []
    step = max(chunk_size - overlap, 1) 

    for i, start in enumerate(range(0, len(words), step)):
        piece = " ".join(words[start:start + chunk_size])
        if piece.strip():
            chunks.append(Chunk(piece, source_id, section, i))

    return chunks


def load_and_chunk_directory(raw_dir: str = None) -> list[Chunk]:
   
    raw_dir = raw_dir or config.RAW_FILINGS_DIR
    all_chunks = []

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"'{raw_dir}' not found. Create it and add plain-text filings first."
        )

    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith(".txt"):
            continue
        source_id = os.path.splitext(filename)[0]
        with open(os.path.join(raw_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(chunk_document(text, source_id=source_id))

    return all_chunks


def save_chunks(chunks: list[Chunk], path: str = "data/chunks.json"):
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2)


if __name__ == "__main__":
    chunks = load_and_chunk_directory()
    print(f"Chunked {len(chunks)} pieces from '{config.RAW_FILINGS_DIR}'")
    save_chunks(chunks)
    print("Saved to data/chunks.json")