import os
import nltk
from nltk.corpus import stopwords
import re

stop_words = set(stopwords.words('english'))

def tokenize_and_filter(query):
    """Tokenize a query and remove stop words."""
    filtered_tokens = {
        token for token in re.findall(r'\w+', query.lower())
        if token not in stop_words and len(token) > 2 and token != "patients"
    }
    return list(filtered_tokens)

def convert_queries_to_boolean(input_file):
    boolean_queries = {}

    with open(input_file, 'r', encoding='utf-8') as file:
        queries = file.readlines()

    for i, query in enumerate(queries, start=1):
        query = query.strip()

        # Tokenize and filter stop words
        terms = tokenize_and_filter(query)

        #  AND for Query 13, OR for all others
        if i == 12:
            boolean_query = " AND ".join(terms)
        else:
            boolean_query = " OR ".join(terms)

        # Store the Boolean query in a dictionary
        boolean_queries[i] = boolean_query

    return boolean_queries

def generate_boolean_queries(output_file):
   
    queries_file = "../collection/Queries.txt"
    boolean_queries = convert_queries_to_boolean(queries_file)

    with open(output_file, 'w', encoding='utf-8') as file:
        for query_id, boolean_query in boolean_queries.items():
            file.write(f"Query {query_id}: {boolean_query}\n")

    print(f"Boolean queries have been written to {output_file}")

if __name__ == "__main__":
    output_file = "../collection/Boolean_Queries.txt"
    generate_boolean_queries(output_file)