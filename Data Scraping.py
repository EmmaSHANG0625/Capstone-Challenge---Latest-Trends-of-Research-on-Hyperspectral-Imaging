import requests
import xml.etree.ElementTree as ET
import pandas as pd

# Define your query parameters
query = "hyperspectral"
max_results = 2600  # Increase this to get more results
url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

# Fetch the data
response = requests.get(url)
data = response.text

# Parse the XML response and extract data
entries = []
root = ET.fromstring(data)
for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
    title = entry.find("{http://www.w3.org/2005/Atom}title").text
    abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text
    authors = [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
    
    # Get the last author
    last_author = authors[-1] if len(authors) > 0 else "N/A"

 
    published_date = entry.find("{http://www.w3.org/2005/Atom}published").text
    
    
    # Append each entry as a dictionary
    entries.append({
        "Title": title,
        "Abstract": abstract,
        "Authors": ", ".join(authors),
        "Last Author": last_author,
        "Published Date": published_date
    })

# Convert to a DataFrame
df = pd.DataFrame(entries)

# Save DataFrame to CSV
df.to_csv("arxiv_hyperspectral_imaging.csv", index=False, encoding="utf-8")
print("Data saved to arxiv_hyperspectral_imaging.csv")


