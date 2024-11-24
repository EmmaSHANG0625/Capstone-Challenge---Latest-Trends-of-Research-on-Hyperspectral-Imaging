import pandas as pd
import spacy
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle

# Load your CSV file
df = pd.read_csv("arxiv_hyperspectral_imaging.csv")

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Define custom stopwords
basic_stopwords = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now"
]
custom_stopwords = basic_stopwords + ["data", "information", "research", "result", "method", "hyperspectral", "image", "imaging", "hsi", "spectral", "use", "study", "approach", "paper", "propose", "base", "algorithm", "datum", "dataset", "model", "result", "state", "art", "problem", "analysis", "work", "different", "technique", "provide", "present", "different", "introduce", "include", "performance", "perform", "proposed", "develop", "application", "evaluate", "compare", "achieve", "show", "apply", "consider", "investigate", "discuss", "sample", "obtain", "demonstrate", "experiment", "new", "novel", "https", "github", "com"]

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc 
              if not token.is_stop and not token.is_punct and token.lemma_.lower() not in custom_stopwords]
    return " ".join(tokens)

# Preprocess the Abstract column
df['cleaned_text'] = df['Abstract'].apply(preprocess_text)

# Save cleaned data to CSV
#df.to_csv("processed_arxiv_data.csv", index=False)

# Word frequency analysis
all_words = ' '.join(df['cleaned_text']).split()
word_freq = Counter(all_words)

# Create bigrams for analysis
vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words='english')
X2 = vectorizer.fit_transform(df['cleaned_text'])
bigram_counts = X2.sum(axis=0).A1
bigrams = vectorizer.get_feature_names_out()
bigram_freq = dict(zip(bigrams, bigram_counts))

# LDA Topic Modeling
vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_text'])

lda = LatentDirichletAllocation(n_components=12, random_state=42)
lda.fit(X)

# Extract LDA topics
lda_topics = [
    [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[::-1][:30]]
    for topic in lda.components_
]

# Save preprocessed data
with open("preprocessed_data.pkl", "wb") as f:
    pickle.dump({
        "all_words": all_words,
        "bigram_freq": bigram_freq,
        "lda_topics": lda_topics,
        "df": df
    }, f)
print("Data preprocessing complete and saved!")
