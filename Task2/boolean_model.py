from collections import deque
import json
import os
import re
from dotenv import load_dotenv


def infix_to_postfix(query):
    """
    Convert an infix Boolean query to postfix notation.
    
    Args:
    - query (str): The Boolean query in infix form.

    Returns:
    - list: The query in postfix form as a list of tokens.
    """
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
    """
    Perform Boolean retrieval using a stack and collections.deque.

    Args:
    - query (str): The Boolean query (e.g., "term1 AND NOT term2").
    - inverted_index (dict): Boolean inverted index.
    - all_docs (set): A set of all document IDs.

    Returns:
    - set: A set of document IDs that satisfy the query.
    """

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

    load_dotenv()

    docs_folder = os.getenv("DOCS_FOLDER")

    if not docs_folder:
        raise ValueError("DOCS_FOLDER environment variable is not set!")

    result = boolean_retrieval_model(postfix_query, boolean_index, docs_folder)
    print(f"Result: {result}")