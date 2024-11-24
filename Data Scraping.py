import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime

# Scrape data and save it to a CSV file
def scrape_arxiv_data():
    query = "hyperspectral"
    max_results = 2600
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    # Fetch the data
    response = requests.get(url)
    data = response.text

    # Parse the XML response
    entries = []
    root = ET.fromstring(data)
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text
        authors = [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
        last_author = authors[-1] if authors else "N/A"
        published_date = entry.find("{http://www.w3.org/2005/Atom}published").text

        entries.append({
            "Title": title,
            "Abstract": abstract,
            "Authors": ", ".join(authors),
            "Last Author": last_author,
            "Published Date": published_date
        })

    # Convert to a DataFrame and save to CSV
    df = pd.DataFrame(entries)
    df.to_csv("arxiv_hyperspectral_imaging.csv", index=False, encoding="utf-8")
    print(f"Data saved to arxiv_hyperspectral_imaging.csv at {datetime.now()}")

# Example: Using Python's schedule module to run every 3 months
if __name__ == "__main__":
    scrape_arxiv_data()
