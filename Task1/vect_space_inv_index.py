import os
from collections import defaultdict
import math
import json
import nltk
from nltk.corpus import stopwords

stopwords_set = set(stopwords.words('english'))

def create_vsm_inverted_index(docs_path):
   
    # Initialize structures
    term_doc_frequency = defaultdict(int)  # Document frequency for each term
    doc_term_frequencies = defaultdict(lambda: defaultdict(int))  # Term frequencies for each doc
    total_documents = 0

    # Parse documents and count term frequencies
    for filename in os.listdir(docs_path):
        total_documents += 1
        doc_id = os.path.splitext(filename)[0]
        filepath = os.path.join(docs_path, filename)

        # Read the document
        with open(filepath, 'r', encoding='utf-8') as file:
            terms = file.read().splitlines()

        # Count term frequencies in the document
        for term in terms:
            term = term.strip()
            if term and term not in stopwords_set and len(term) > 2:
                doc_term_frequencies[doc_id][term] += 1

        # Update document frequency for terms
        for term in set(doc_term_frequencies[doc_id].keys()):
            term_doc_frequency[term] += 1

    # Compute TF-IDF for each term in each document
    vsm_inverted_index = defaultdict(dict)

    for doc_id, term_frequencies in doc_term_frequencies.items():
       # Compute the maximum frequency of any term in the query
        max_freq = max(term_frequencies.values(), default=1)

        for term, tf in term_frequencies.items():
            tf_value = tf /max_freq

            # Calculate IDF
            idf_value = math.log2(total_documents / (1 + term_doc_frequency[term]))

            # Calculate TF-IDF
            tf_idf_value = tf_value * idf_value

            # Update the VSM inverted index
            vsm_inverted_index[term][doc_id] = tf_idf_value

    return vsm_inverted_index

def save_vsm_inverted_index(vsm_inverted_index, output_path):
   
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vsm_inverted_index, f, indent=4)


if __name__ == "__main__":
    docs_folder = "../collection/docs"
    
    vsm_index = create_vsm_inverted_index(docs_folder)

    save_vsm_inverted_index(vsm_index, "vsm_inverted_index.json")
    print("VSM Inverted Index saved as 'vsm_inverted_index.json'")