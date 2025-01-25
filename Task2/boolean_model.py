from collections import deque
import json
import os
import re


def infix_to_postfix(query):
   
    precedence = {"NOT": 3, "AND": 2, "OR": 1}
    output = []
    operators = deque()
    tokens = re.findall(r'\(|\)|\w+|AND|OR|NOT', query.upper())

    for token in tokens:
        if token in precedence:  # If the token is an operator
            while (operators and operators[-1] != "(" and
                   precedence[token] <= precedence[operators[-1]]):
                output.append(operators.pop())
            operators.append(token)
        elif token == "(":  # Left parenthesis
            operators.append(token)
        elif token == ")":  # Right parenthesis
            while operators and operators[-1] != "(":
                output.append(operators.pop())
            operators.pop()  # Remove the "("
        else:  # If the token is a term
            output.append(token)
    
    while operators:
        output.append(operators.pop())
    
    return output


def boolean_retrieval_model(query, inverted_index, all_docs):

    # Initialize an empty stack using deque
    stack = deque()

    # Process the tokens
    for token in query:
        if token == "AND":
            # Pop the last two sets and calculate their intersection
            set2 = stack.pop()
            set1 = stack.pop()
            stack.append(set1 & set2)
        elif token == "OR":
            # Pop the last two sets and calculate their union
            set2 = stack.pop()
            set1 = stack.pop()
            stack.append(set1 | set2)
        elif token == "NOT":
            # Pop the last set and calculate the complement
            set1 = stack.pop()
            stack.append(all_docs - set1)
        else:
            # Push the document set for the term onto the stack
            # If the term is not in the inverted index, push an empty set
            stack.append(inverted_index.get(token.lower(), set()))

    # The final result should be the only item remaining in the stack
    return stack.pop() if stack else set()


if __name__ == "__main__":
    
        # Load the Boolean inverted index from the JSON file
    index_file_path = "../Task1/boolean_index.json"
    if os.path.exists(index_file_path):
        with open(index_file_path, 'r', encoding='utf-8') as f:
            boolean_index = json.load(f)

            # Convert all lists back into sets
        boolean_index = {term: set(doc_ids) for term, doc_ids in boolean_index.items()}
    else:
        raise FileNotFoundError(f"Index file not found: {index_file_path}")

   
    # Ask user for query input
    query = input("Enter your Boolean query: ")
    print(f"Query: {query}")

    postfix_query = infix_to_postfix(query)
    print(f"Postfix Query: {postfix_query}")

    docs_folder = "../collection/docs"

     
    # Perform the Boolean retrieval   
    result_ids = boolean_retrieval_model(postfix_query, boolean_index, set(os.listdir(docs_folder)))
    print(f"Matching Document IDs: {result_ids}")

    # Store matching document contents in a file
    output_file = "bool_retrieved_docs.txt"
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for doc_id in result_ids:
            doc_path = os.path.join(docs_folder, f"{doc_id}")
            if os.path.exists(doc_path):
                with open(doc_path, 'r', encoding='utf-8') as doc_file:
                    content = doc_file.read()
                    out_file.write(f"--- Document {doc_id} ---\n")
                    out_file.write(content)
                    out_file.write("\n\n")
            else:
                print(f"Document file for ID {doc_id} not found in {docs_folder}!")

    print(f"Matching documents have been saved to {output_file}")