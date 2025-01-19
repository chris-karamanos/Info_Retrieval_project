import math
import os
import json
import re
from collections import defaultdict, Counter
from nltk.corpus import stopwords

stopwords_set = set(stopwords.words('english'))

def load_vsm_inverted_index(index_path):
    
    with open(index_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_queries(queries_path):
    
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
   
    dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) for term in query_vector)

    query_magnitude = math.sqrt(sum(weight**2 for weight in query_vector.values()))
    doc_magnitude = math.sqrt(sum(weight**2 for weight in doc_vector.values()))

    # Avoid division by zero
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0.0

    return dot_product / (query_magnitude * doc_magnitude)

def save_retrieved_docs(doc_ids, docs_folder, output_file):
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for doc_id in doc_ids:
            doc_path = os.path.join(docs_folder, f"{doc_id}")
            if os.path.exists(doc_path):
                with open(doc_path, 'r', encoding='utf-8') as doc_file:
                    content = doc_file.read()
                    out_file.write(f"--- Document {doc_id} ---\n")
                    out_file.write(content)
                    out_file.write("\n\n")
            else:
                print(f"Document file for ID {doc_id} not found in {docs_folder}!")

if __name__ == "__main__":
    # Step 1: Load the VSM Inverted Index
    vsm_index_path = "../Task1/vsm_inverted_index.json"
    vsm_inverted_index = load_vsm_inverted_index(vsm_index_path)

    # Step 2: Load and Parse Queries
    queries_path = "../collection/Queries.txt"
    queries = load_queries(queries_path)
    
    # Total number of documents
    total_documents = len(set(doc_id for term_docs in vsm_inverted_index.values() for doc_id in term_docs))

    # Step 3 & 4: Process each query and rank documents
    for query_id, query_text in queries.items():
        print(f"\nProcessing Query {query_id}: {query_text}")
        
        # Compute TF-IDF vector for the query
        query_vector = process_query(query_text, vsm_inverted_index, total_documents)
        
        # Rank documents based on cosine similarity
        ranked_docs = rank_documents(query_vector, vsm_inverted_index)
        
        print(f"Top Ranked Documents for Query {query_id}:")
        for doc_id, score in ranked_docs[:10]:  # Show top 10 documents
            print(f"  Document {doc_id}: {score:.4f}")

        # Get top 5 ranked documents
        top_docs = [doc_id for doc_id, _ in ranked_docs[:5]]   

        # Save the contents of the top documents to a file
        docs_folder = "../collection/docs"
        output_file = f"vsm_retrieved_docs/vsm_retrieved_docs_query_{query_id}.txt"
        save_retrieved_docs(top_docs, docs_folder, output_file)     