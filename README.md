### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 24/11/2026
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:
```
import nltk
import string
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from tabulate import tabulate

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?"
}

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words("english") and word not in string.punctuation]
    return tokens

# Preprocess documents
preprocessed_docs = {
    doc_id: preprocess_text(doc)
    for doc_id, doc in documents.items()
}

# Find all unique terms
terms = sorted(set(word for words in preprocessed_docs.values() for word in words))

N = len(documents)

# Calculate Term Frequency (TF)
tf = {}

for doc_id, words in preprocessed_docs.items():
    counts = Counter(words)
    tf[doc_id] = {term: counts[term] for term in terms}

print("\n--- Term Frequencies (TF) ---\n")

tf_table = []

for doc_id in documents:
    tf_table.append([doc_id] + [tf[doc_id][term] for term in terms])

print(tabulate(
    [["Doc ID"] + terms] + tf_table,
    tablefmt="grid",
    headers="firstrow"
))

# Calculate Document Frequency and IDF
df = {}
idf = {}

for term in terms:
    df[term] = sum(1 for doc_id in documents if tf[doc_id][term] > 0)
    idf[term] = math.log10(N / df[term]) + 1

print("\n--- Document Frequency (DF) and Inverse Document Frequency (IDF) ---\n")

df_table = [
    [term, df[term], round(idf[term], 4)]
    for term in terms
]

print(tabulate(
    df_table,
    headers=["Term", "Document Frequency (DF)", "Inverse Document Frequency (IDF)"],
    tablefmt="grid"
))

# Calculate TF-IDF weights
tfidf = {}

for doc_id in documents:
    tfidf[doc_id] = {
        term: tf[doc_id][term] * idf[term]
        for term in terms
    }

print("\n--- TF-IDF Weights ---\n")

tfidf_table = []

for doc_id in documents:
    tfidf_table.append([
        doc_id
    ] + [
        round(tfidf[doc_id][term], 4)
        for term in terms
    ])

print(tabulate(
    [["Doc ID"] + terms] + tfidf_table,
    tablefmt="grid",
    headers="firstrow"
))

# Search function
def search(query, tfidf_matrix, tfidf_vectorizer):

    query_words = preprocess_text(query)
    query_counts = Counter(query_words)

    query_tfidf = {}

    for term in terms:
        query_tfidf[term] = query_counts[term] * idf[term]

    # Calculate cosine similarity
    results = []

    query_magnitude = math.sqrt(
        sum(value ** 2 for value in query_tfidf.values())
    )

    for doc_id, doc in documents.items():

        dot_product = sum(
            query_tfidf[term] * tfidf[doc_id][term]
            for term in terms
        )

        doc_magnitude = math.sqrt(
            sum(value ** 2 for value in tfidf[doc_id].values())
        )

        if query_magnitude == 0 or doc_magnitude == 0:
            similarity = 0
        else:
            similarity = dot_product / (query_magnitude * doc_magnitude)

        results.append((
            doc_id,
            doc,
            dot_product,
            query_magnitude,
            doc_magnitude,
            round(similarity, 4)
        ))

    results.sort(key=lambda x: x[5], reverse=True)

    return results, query_tfidf


# Get input from user
query = input("\nEnter your query: ")

# Perform search
search_results, query_tfidf = search(query, tfidf, None)

print("\n--- Search Results and Cosine Similarity ---\n")

result_table = []

for result in search_results:
    result_table.append([
        result[0],
        result[1],
        round(result[2], 4),
        round(result[3], 4),
        round(result[4], 4),
        result[5]
    ])

print(tabulate(
    result_table,
    headers=[
        "Doc ID",
        "Document",
        "Dot Product",
        "Query Magnitude",
        "Doc Magnitude",
        "Cosine Similarity"
    ],
    tablefmt="grid"
))

print("\n--- Query TF-IDF Weights ---\n")

query_table = [
    [term, round(weight, 4)]
    for term, weight in query_tfidf.items()
    if weight > 0
]

print(tabulate(
    query_table,
    headers=["Term", "Query TF-IDF Weight"],
    tablefmt="grid"
))

print("\n--- Ranked Documents ---\n")

ranked_table = []

for rank, result in enumerate(search_results, start=1):
    ranked_table.append([
        rank,
        result[0],
        result[1],
        result[5]
    ])

print(tabulate(
    ranked_table,
    headers=["Rank", "Document ID", "Document Text", "Cosine Similarity"],
    tablefmt="grid"
))

highest_rank_score = search_results[0][5]

print(
    "\nThe highest rank cosine score is:",
    highest_rank_score,
    "(Document ID:",
    search_results[0][0] + ")"
)
```

### Output:

<img width="843" height="802" alt="image" src="https://github.com/user-attachments/assets/59ec58d6-68b0-4517-b51b-699bbf7c1554" />
<img width="1056" height="652" alt="image" src="https://github.com/user-attachments/assets/a6122252-aa2a-4157-83e2-69c47d438f7f" />
<img width="1097" height="681" alt="image" src="https://github.com/user-attachments/assets/cdecbffa-3edc-438d-a995-684d55d6c627" />

### Result:
Thus, Retrieval Using Vector Space Model in Python is successfully verified.

