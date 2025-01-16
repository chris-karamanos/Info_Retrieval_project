import os
import json
import sys

# Add Task3 folder to Python's module search path
sys.path.append(os.path.abspath("../Task3"))
from vector_space_model import process_query, rank_documents

def load_relevant_documents(relevant_file):
    """
    Load relevant documents from Relevant.txt into a dictionary.

    Args:
    - relevant_file (str): Path to the Relevant.txt file.

    Returns:
    - dict: Dictionary with query IDs as keys and sets of relevant document IDs as values.
    """
    relevant_docs = {}
    with open(relevant_file, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, start=1):
            relevant_docs[i] = set(line.strip().split())
    return relevant_docs

def compute_incremental_metrics_vsm(query_file, relevant_file, index_file):
    """
    Compute precision scores incrementally for the top 10 documents retrieved by the Vector Space Model.

    Args:
    - query_file (str): Path to the file containing queries.
    - relevant_file (str): Path to the file containing relevant document IDs.
    - index_file (str): Path to the inverted index JSON file with TF-IDF values.

    Returns:
    - dict: Incremental precision scores for each query.
    """
    # Load queries
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = {i + 1: line.strip() for i, line in enumerate(f)}

    # Load relevant documents
    relevant_docs = load_relevant_documents(relevant_file)

    # Load inverted index
    with open(index_file, 'r', encoding='utf-8') as f:
        inverted_index = json.load(f)

    # Calculate total number of documents
    total_documents = len({doc_id for term_docs in inverted_index.values() for doc_id in term_docs})

    # Compute precision and recall incrementally for each query
    incremental_metrics_scores = {}
    for query_id, query in queries.items():
        # Process query to get its vector representation
        query_vector = process_query(query, inverted_index, total_documents)

        # Rank documents using cosine similarity (top 10 documents)
        ranked_docs = rank_documents(query_vector, inverted_index)[:10]

        # Retrieve documents and calculate incremental precision
        retrieved_docs = []
        relevant = relevant_docs.get(query_id, set())
        precision_at_k = []
        recall_at_k = []

        for doc_id, _ in ranked_docs:
            # Strip leading zeros and add to retrieved list
            retrieved_docs.append(doc_id.lstrip("0"))

            # Calculate precision for current k
            relevant_retrieved = set(retrieved_docs).intersection(relevant)
            precision = len(relevant_retrieved) / len(retrieved_docs) if len(retrieved_docs) > 0 else 0.0
            recall = len(relevant_retrieved) / len(relevant) if len(relevant) > 0 else 0.0
            precision_at_k.append(precision)
            recall_at_k.append(recall)


        # Print retrieved documents
        print(f"Query {query_id}: Retrieved documents = {retrieved_docs}")

        # Save incremental precision and recall scores for this query
        incremental_metrics_scores[query_id] = {
            "precision": precision_at_k,
            "recall": recall_at_k
        }

    return incremental_metrics_scores

# Example usage
if __name__ == "__main__":
    query_file = "../collection/Queries.txt"
    relevant_file = "../collection/Relevant.txt"
    index_file = "../Task1/vsm_inverted_index.json"

    incremental_metrics_scores = compute_incremental_metrics_vsm(query_file, relevant_file, index_file)

    # Print incremental precision scores
    for query_id, metrics in incremental_metrics_scores.items():
        print(f"Query {query_id}:")
        for k, (precision,recall) in enumerate(zip(metrics["precision"],metrics["recall"]), start=1):
            print(f"  Precision@{k}: {precision:.4f}, Recall@{k}: {recall:.4f}")