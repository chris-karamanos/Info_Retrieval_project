import time
import os
import sys
import json

# Add Task1, Task2 and Task3 folders to Python's module search path
sys.path.append(os.path.abspath("../Task1"))  # For indexing functions
sys.path.append(os.path.abspath("../Task2"))
sys.path.append(os.path.abspath("../Task3"))

from boolean_inverted_index import create_boolean_inverted_index
from vect_space_inv_index import create_vsm_inverted_index
from boolean_model import infix_to_postfix, boolean_retrieval_model
from vector_space_model import process_query, rank_documents

def load_relevant_documents(relevant_file):
    
    relevant_docs = {}
    with open(relevant_file, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, start=1):
            relevant_docs[i] = set(line.strip().split())
    return relevant_docs

def measure_indexing_time(docs_folder, index_type="boolean"):
  
    # Start timing
    start_time = time.time()
    
    if index_type == "boolean":
        create_boolean_inverted_index(docs_folder)
    elif index_type == "vsm":
        create_vsm_inverted_index(docs_folder)
    else:
        raise ValueError("Invalid index_type. Use 'boolean' or 'vsm'.")
    
    # End timing
    end_time = time.time()
    return end_time - start_time

def run_boolean_model(query_file, index_file, docs_folder):
   
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = {i + 1: line.strip() for i, line in enumerate(f)}

    with open(index_file, 'r', encoding='utf-8') as f:
        boolean_index = json.load(f)

    boolean_index = {term: set(doc_ids) for term, doc_ids in boolean_index.items()}

    all_docs = set(os.listdir(docs_folder))

    start_time = time.time()

    for query_id, query in queries.items():
        postfix_query = infix_to_postfix(query)
        boolean_retrieval_model(postfix_query, boolean_index, all_docs)

    end_time = time.time()

    return end_time - start_time

def run_vector_space_model(query_file, index_file):
   
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = {i + 1: line.strip() for i, line in enumerate(f)}

    with open(index_file, 'r', encoding='utf-8') as f:
        inverted_index = json.load(f)

    total_documents = len({doc_id for term_docs in inverted_index.values() for doc_id in term_docs})

    start_time = time.time()

    for query_id, query in queries.items():
        query_vector = process_query(query, inverted_index, total_documents)
        rank_documents(query_vector, inverted_index)

    end_time = time.time()

    return end_time - start_time

if __name__ == "__main__":
    # Define file paths
    boolean_query_file = "../collection/Boolean_Queries.txt"
    vsm_query_file = "../collection/Queries.txt"
    relevant_file = "../collection/Relevant.txt"
    boolean_index_file = "../Task1/boolean_index.json"
    vsm_index_file = "../Task1/vsm_inverted_index.json"
    docs_folder = "../collection/docs"

    boolean_indexing_time = measure_indexing_time(docs_folder, index_type="boolean")
    print(f"Boolean Indexing Time: {boolean_indexing_time:.4f} seconds")

    vsm_indexing_time = measure_indexing_time(docs_folder, index_type="vsm")
    print(f"Vector Space Model Indexing Time: {vsm_indexing_time:.4f} seconds")

    boolean_execution_time = run_boolean_model(boolean_query_file, boolean_index_file, docs_folder)
    print(f"Boolean Model Execution Time: {boolean_execution_time:.4f} seconds")

    vsm_execution_time = run_vector_space_model(vsm_query_file, vsm_index_file)
    print(f"Vector Space Model Execution Time: {vsm_execution_time:.4f} seconds")