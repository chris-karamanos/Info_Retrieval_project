import os
from collections import defaultdict
import json
import nltk
from nltk.corpus import stopwords

stopwords_set = set(stopwords.words('english'))

def create_boolean_inverted_index(docs_path):
    """
    Create a Boolean inverted index from a collection of documents.

    Args:
    - docs_path (str): Path to the folder containing the documents.

    Returns:
    - defaultdict: Inverted index mapping terms to sets of document IDs.
    """
    # Initialize an empty defaultdict for the inverted index
    inverted_index = defaultdict(set)
    
    # Loop through all files in the directory
    for filename in os.listdir(docs_path):
        # Get document ID from the filename (assuming filenames are like '00001', '00002', etc.)
        doc_id = os.path.splitext(filename)[0]
        
        # Construct the full file path
        filepath = os.path.join(docs_path, filename)
        
        # Open and read the document content
        with open(filepath, 'r', encoding='utf-8') as file:
            terms = file.read().splitlines()  # Each line is a token (term)
        
        # Update the inverted index
        for term in terms:
            term = term.strip()  # Clean up the term
            if term and term not in stopwords_set and len(term)>2:  # Ensure the term is not empty
                inverted_index[term].add(doc_id)
    
    return inverted_index

def save_inverted_index(inverted_index, output_path):
    """
    Save the inverted index to a file in JSON format.

    Args:
    - inverted_index (dict): The inverted index to save.
    - output_path (str): Path to the output file.
    """
    # Convert sets to lists for JSON compatibility
    json_compatible_index = {term: list(doc_ids) for term, doc_ids in inverted_index.items()}
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_compatible_index, f, indent=4)

# Example usage:
if __name__ == "__main__":
    # Path to the folder containing the documents
    docs_folder = r"C:\Users\chrka\OneDrive\Documents\πανεπιστημιο\11ο εξάμηνο\Ανάκτηση Πληροφορίας\Ανάκτηση Πληροφορίας 2024-2025\collection\docs"

    # Create the Boolean inverted index
    boolean_index = create_boolean_inverted_index(docs_folder)
    
    # Save the index to a file
    save_inverted_index(boolean_index, "boolean_index.json")
    print("Inverted index saved as 'boolean_index.json'")
