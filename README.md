# RAG PDF Q&A with Google Gemini
This project is a fully functional, self-contained web application that allows users to chat with their PDF documents. It implements the Retrieval-Augmented Generation (RAG) pattern to provide accurate, context-aware answers based on the content of an uploaded PDF.

The user can upload a PDF, which is then processed, chunked, and stored in a vector database. When a user asks a question, the application retrieves the most relevant text chunks from the document and uses Google's Gemini Pro model to generate a natural language answer based solely on that context.

<img width="622" height="651" alt="Screenshot 2025-09-08 at 15 28 12" src="https://github.com/user-attachments/assets/7090f0ff-a445-4133-bfb5-b3a71b2f2b1d" />


# Features
Easy PDF Upload: A simple web interface to upload any PDF document.

Dynamic Document Collections: Users can name their document collections, allowing for multiple PDFs to be stored and queried independently.

Intelligent Q&A: Leverages a powerful sentence-transformer model for embeddings and Google's Gemini for answer synthesis.

Context-Aware Answers: The model is instructed to answer questions only based on the information present in the PDF, reducing hallucinations.

Source Citing: Displays which page of the document the answer was derived from, along with a similarity score.

Self-Contained & Local: The entire application (frontend and backend) runs locally with a single command, making it easy to test and demonstrate.

# How It Works: The RAG Pipeline
The application follows a classic Retrieval-Augmented Generation architecture:

Ingestion: The uploaded PDF is parsed, and its text content is extracted page by page.

Chunking: The extracted text is split into smaller, overlapping chunks to ensure semantic continuity.

Embedding & Storage: Each text chunk is converted into a numerical vector (an embedding) using the all-mpnet-base-v2 model and stored in a ChromaDB vector database.

Retrieval: When a user asks a question, their query is also embedded. The application then performs a similarity search in the vector database to find the text chunks most relevant to the question.

Generation: The retrieved text chunks (the context) and the user's original question are passed to the Google Gemini model. The model is prompted to synthesize a final answer based only on the provided context.

# Tech Stack

### Backend Framework

Python, Flask

### Language Model

Google Gemini 2.0 Flash

### Vector Database

ChromaDB (Persistent)

### Embedding Model

sentence-transformers/all-mpnet-base-v2

### PDF Parsing

PyMuPDF

### Frontend

HTML, CSS, Vanilla JavaScript (Single-Page App)

### Dependencies

python-dotenv, numpy

# How to Run Locally
Follow these steps to set up and run the project on your local machine.

Prerequisites
Python 3.9+

An API Key for the Google Gemini API. You can get one from Google AI Studio.

### 1. Clone the Repository

First, clone this repository to your local machine.

git clone [https://github.com/YOUR_USERNAME/rag-pdf-qa.git](https://github.com/YOUR_USERNAME/rag-pdf-qa.git)

cd rag-pdf-qa

### 2. Set Up a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

#### Create a virtual environment
python3 -m venv .venv

## Activate it
### On macOS/Linux:
source .venv/bin/activate
### On Windows:
.\.venv\Scripts\activate
##

### 3. Install Dependencies
Install all the required packages using the requirements.txt file.

pip install -r requirements.txt

### 4. Configure Your API Key
You need to provide your Gemini API key to the application.

Create a new file named .env in the root of the project directory.

Add your API key to this file in the following format:

GEMINI_API_KEY="YOUR_API_KEY_HERE"

This .env file is listed in .gitignore, so your secret key will not be committed to GitHub.

### 5. Run the Application
Launch the Flask development server with a single command.

flask run

The application will now be running. Open your web browser and navigate to:

http://127.0.0.1:5000/

You can now upload a PDF and start asking questions!
