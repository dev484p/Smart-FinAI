# SMART FIN-AI

SMART FIN-AI is an advanced chatbot designed to assist with finance-related queries. It leverages a Retrieval-Augmented Generation (RAG) pipeline to deliver accurate and relevant answers by integrating Chroma DB for efficient data storage and retrieval, and HuggingFace embeddings for improved natural language understanding.

## 🚀 Features
* Contextual Financial Responses: Provides accurate answers based on finance-related documents and knowledge.
* Advanced RAG Pipeline: Utilizes LangChain's RAG for retrieval and response generation.
* Efficient Embeddings: Employs HuggingFace embeddings for improved understanding and semantic matching.
* Chroma DB Integration: Stores and retrieves financial knowledge with Chroma DB for high-performance querying.
* Customizable & Scalable: Easily adaptable to various finance use cases and extensible to support more data sources.
* Powered by Llama2: Utilizes the robust capabilities of Llama2 for generative responses and superior performance.

## 🛠️ Tech Stack
* LangChain: Framework for building applications powered by language models.
* Chroma DB: Vector database for efficient data retrieval.
* HuggingFace Models: State-of-the-art pre-trained embeddings for natural language processing.
* Python: Primary language for development.
* llama2*: State-of-the-art generative AI model.
 
## 🚀 Getting Started
### Prerequisites
* Python 3.8 or later
* pip (Python package manager)

### Installation
```
pip install -r requirements.txt
```
### Note - Make sure you have installed Ollama and Lamma2 :
Ollama is an AI library designed to work with various LLM models. You need to:
* Download and set up Ollama: Follow the [Ollama documentation](https://ollama.com/download/windows) to install the runtime and configure it on your machine.
* After setup of Ollama run following to install Lamma2
  ```
  ollama run llama2
  ```
