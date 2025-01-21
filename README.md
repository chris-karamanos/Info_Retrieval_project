# Info_Retrieval_project

## Overview

This repository contains the implementation of an Information Retrieval system based on the requirements outlined in the Information Retrieval course at the University of Patras, Department of Computer Engineering and Informatics. The project includes the study, implementation, and comparison of two retrieval models:

- Boolean Model
- Vector Space Model (VSM)

The project also involves evaluation using precision, recall, and execution time metrics, and concludes with the development of a chatbot leveraging retrieval results (using frameworks like LangChain and OLLAMA).

## Repository Structure

- **Task1/:**
  - **boolean_inverted_index.py**: Generates the Boolean Inverted index.
  - **vect_space_inv_index**: Generates the Vector Space Model (VSM) inverted index using TF-IDF.

- **Task2/:**
  - **boolean_model.py:** Implements Boolean query processing using the inverted index.
- **Task3/:**
  - **vector_space_model.py:** Implements the Vector Space Model, including query processing, TF-IDF computation, and document ranking based on cosine similarity.
- **Task4/:**
  - **generate_boolean_queries.py:** Generates Boolean queries and saves them to Boolean_Queries.txt.
  - **boolean_metrics.py:** Computes precision and recall for the Boolean model.
  - **vect_metrics.py:** Computes precision and recall for the Vector Space Model.
  - **exec_time.py:** Measures the execution time of both the Boolean and Vector Space models, including indexing.
- **Task5/:**
  - **chatbot.py:** Interactive interface built to explore and query the results of the Boolean and Vector Space Model (VSM) information retrieval methods. Users can interact with documents retrieved by these models through natural language queries.
- **collection/:**
  - **Queries.txt:** Natural language queries.
  - **Boolean_Queries.txt:** Boolean queries generated from Queries.txt.
  - **docs:** The collection of documents to be indexed and retrieved.
  - **Relevant.txt:** Relevant documents IDs for the provided queries.
 

## Installation

1. Clone the repository
``` bash
git clone git@github.com:chris-karamanos/Info_Retrieval_project.git
```
2. Navigate to the project directory:
``` bash
cd Info_Retrieval_project
```
3. Ensure python is downloaded
``` bash
python3 --version
```

4. Ensure NLTK stopwords are downloaded
``` bash
import ntlk
ntlk.download('stopwords')
```
### For the chatbot:
1. **Install Ollama:** 
- Ollama is required to serve the language model. Download and install Ollama following the official instructions from the [Ollama GitHub Repository](https://github.com/ollama). 
- Ensure that the Ollama service is running on your system. You can start the service using the following command:
     ```bash
     ollama serve
     ```

2. **Choose a Suitable Language Model**
  - Select a model based on your system's memory and computational capabilities.
  - **Note:** Choosing a model with high memory requirements may lead to performance issues. Ensure that your system meets the resource requirements of the selected model.
   - Once the Ollama service is running in the background, download the model using the following command:
     ```bash
     ollama pull <model_name>
     ```
   - For this project, the **`phi`** model is recommended:
     ```bash
     ollama pull phi
     ```

3. **Install Required Python Libraries (LangChain Libraries):** Install the following dependencies using pip (or pip3):
```bash
pip install langchain
pip install faiss-cpu  # Use `faiss-gpu` if you have a GPU
pip install langchain-ollama
pip install langchain-huggingface
pip install langchain-core
```






# Evaluation
- **Precision and Recall:** Evaluated at incremental retrieval levels (e.g., Precision@k, Recall@k) for both models.
- **Executon Time:** Time measurements for indexing and query processing.
- **Comparison:** Quantitative and qualitative comparison of Boolean and VSM.

# Challenges
- Handling missing documents.
- Normalizing queries and documents to handle case sensitivity, stop words and punctuation.
- Efficiency implementing and optimizing cosine similarity for large collections.

# Results
The top-ranked results for each query were evaluated, and the metrics are summarized in *Task4* scripts. Execution logs and examples of retrieval outputs are included in the repository for reference.

