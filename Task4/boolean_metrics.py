import os
import json
import sys

# Add Task2 and Task4 folders to Python's module search path
sys.path.append(os.path.abspath("../Task2"))  # For boolean_model.py
sys.path.append(os.path.abspath("."))        # For generate_boolean_queries.py

import boolean_model 

def load_relevant_documents(relevant_file):

    relevant_docs = {}
    with open(relevant_file, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, start=1):
            relevant_docs[i] = set(map(int, line.strip().split()))
    return relevant_docs


def load_boolean_queries(query_file):
   
    boolean_queries = {}
    with open(query_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(":", 1)
            if len(parts) == 2:
                query_id = int(parts[0].split()[1])
                boolean_queries[query_id] = parts[1].strip()
    return boolean_queries

def compute_precision_and_recall(queries, relevant_docs, index_file, docs_folder):
   
    with open(index_file, 'r', encoding='utf-8') as f:
        boolean_index = json.load(f)
    boolean_index = {term: set(doc_ids) for term, doc_ids in boolean_index.items()}

    # Get all document IDs
    all_docs = set(os.listdir(docs_folder))

    # Retrieve documents for each query using the Boolean model
    scores = {}
    for query_id, query in queries.items():
        # Convert the query to postfix notation
        postfix_query = boolean_model.infix_to_postfix(query)
        # Perform Boolean retrieval
        retrieved_docs = boolean_model.boolean_retrieval_model(postfix_query, boolean_index, all_docs)

        # Normalize retrieved_docs to integers by stripping leading zeros
        retrieved_docs = {int(doc_id.lstrip("0")) for doc_id in retrieved_docs}

        relevant = relevant_docs.get(query_id, set())
        relevant_retrieved = retrieved_docs.intersection(relevant)
        precision = len(relevant_retrieved) / len(retrieved_docs) if len(retrieved_docs) > 0 else 0.0
        recall = len(relevant_retrieved) / len(relevant) if len(relevant) > 0 else 0.0
        
        scores[query_id] = {"precision": precision, "recall": recall}

    return scores


if __name__ == "__main__":
    relevant_file = "../collection/Relevant.txt"
    query_file = "../collection/Boolean_Queries.txt"
    index_file = "../Task1/boolean_index.json"
    docs_folder = "../collection/docs"

    relevant_docs = load_relevant_documents(relevant_file)

    queries = load_boolean_queries(query_file)

    scores = compute_precision_and_recall(queries, relevant_docs, index_file, docs_folder)

    for query_id, metrics in scores.items():
        print(f"Query {query_id}: Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}")