import math
import json
import re
from collections import defaultdict, Counter
from nltk.corpus import stopwords

# Define the stop words set
stopwords_set = set(stopwords.words('english'))

def load_vsm_inverted_index(index_path):
    """
    Load the VSM inverted index from a JSON file.

    Args:
    - index_path (str): Path to the VSM inverted index JSON file.

    Returns:
    - dict: The VSM inverted index.
    """
    with open(index_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_queries(queries_path):
    """
    Load and parse queries from Queries.txt.

    Args:
    - queries_path (str): Path to the queries file.

    Returns:
    - dict: A dictionary mapping query IDs to their text.
    """
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:  # Skip empty lines
                print(f"Skipping empty line at query {i}")
                continue
            queries[str(i)] = line  # Assign numeric IDs as strings
    return queries

def process_query(query, vsm_inverted_index, total_documents):
    """
    Process a query and compute its TF-IDF vector.

    Args:
    - query (str): The user's query.
    - vsm_inverted_index (dict): The VSM inverted index.
    - total_documents (int): Total number of documents in the collection.

    Returns:
    - dict: The TF-IDF vector for the query.
    """
    # Tokenize the query and filter out stop words
    tokens = [token for token in re.findall(r'\w+', query.lower()) if token not in stopwords_set and len(token) > 2]
    
    # Count term frequencies in the query
    term_frequencies = Counter(tokens)
    
    # Compute the maximum frequency of any term in the query
    max_freq = max(term_frequencies.values(), default=1)

    # Compute the TF-IDF vector for the query
    query_vector = {}

    for term, tf in term_frequencies.items():
        # Term frequency (normalized by query length)
        tf_value = tf / max_freq

        # Inverse document frequency (use the VSM index to get document frequency)
        idf_value = math.log2(total_documents / (1 + len(vsm_inverted_index.get(term, {}))))
        # TF-IDF calculation
        query_vector[term] = tf_value * idf_value

    return query_vector

def rank_documents(query_vector, vsm_inverted_index):
    """
    Rank documents based on their cosine similarity to the query vector.

    Args:
    - query_vector (dict): TF-IDF vector for the query.
    - vsm_inverted_index (dict): The VSM inverted index.

    Returns:
    - list: A list of tuples (doc_id, similarity_score), sorted by similarity score in descending order.
    """

    document_vectors = defaultdict(dict)

    # Calculate similarity scores for all documents
    for term, query_weight in query_vector.items():
        if term in vsm_inverted_index:
            for doc_id, doc_weight in vsm_inverted_index[term].items():
                document_vectors[doc_id][term] = doc_weight

    # Step 2: Compute cosine similarity for each document
    document_scores = {}
    for doc_id, doc_vector in document_vectors.items():
        document_scores[doc_id] = cosine_similarity(query_vector, doc_vector)

    # Normalize scores by document magnitude
    ranked_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs


def cosine_similarity(query_vector, doc_vector):
    """
    Compute the cosine similarity between two vectors.

    Args:
    - query_vector (dict): TF-IDF vector for the query.
    - doc_vector (dict): TF-IDF vector for a document.

    Returns:
    - float: The cosine similarity score.
    """
    # Compute dot product
    dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) for term in query_vector)

    # Compute magnitudes
    query_magnitude = math.sqrt(sum(weight**2 for weight in query_vector.values()))
    doc_magnitude = math.sqrt(sum(weight**2 for weight in doc_vector.values()))

    # Avoid division by zero
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0.0

    return dot_product / (query_magnitude * doc_magnitude)

if __name__ == "__main__":
    # Step 1: Load the VSM Inverted Index
    vsm_index_path = "../Task1/vsm_inverted_index.json"
    vsm_inverted_index = load_vsm_inverted_index(vsm_index_path)
    print(f"Loaded VSM Inverted Index: {len(vsm_inverted_index)} terms.")

    # Step 2: Load and Parse Queries
    queries_path = "../collection/Queries.txt"
    queries = load_queries(queries_path)
    print(f"Loaded Queries: {len(queries)} queries.")
    
    # Total number of documents
    total_documents = len(set(doc_id for term_docs in vsm_inverted_index.values() for doc_id in term_docs))

    # Step 3 & 4: Process each query and rank documents
    for query_id, query_text in queries.items():
        print(f"\nProcessing Query {query_id}: {query_text}")
        
        # Compute TF-IDF vector for the query
        query_vector = process_query(query_text, vsm_inverted_index, total_documents)
        
        # Rank documents based on cosine similarity
        ranked_docs = rank_documents(query_vector, vsm_inverted_index)
        
        # Display top 10 ranked documents
        print(f"Top Ranked Documents for Query {query_id}:")
        for doc_id, score in ranked_docs[:10]:  # Show top 10 documents
            print(f"  Document {doc_id}: {score:.4f}")
