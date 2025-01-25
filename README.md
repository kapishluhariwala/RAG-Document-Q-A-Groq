# RAG Chatbot using Groq

Welcome to the **RAG Chatbot** repository! This project showcases a **Retrieval-Augmented Generation (RAG)** chatbot powered by **Groq** for high-performance AI processing and **Streamlit** for an intuitive user interface.

---

## Features

- **Retrieval-Augmented Generation:** Combines retrieval of relevant documents with generative AI for precise and context-aware responses.
- **Powered by Groq:** Leverages Groq's high-speed AI processing for low-latency interactions.
- **Streamlit-based Interface:** Provides a user-friendly and interactive UI for seamless communication.

---

## Live Demo

Try out the RAG Chatbot in action using our live demo:

[Access the Chatbot](https://rag-chatbot---groq.streamlit.app/)

Interact with the chatbot, test retrieval and generation capabilities, and experience real-time responses.

---

## How It Works

### RAG Architecture

1. **Retrieval Phase:**
   - The chatbot queries a document database to retrieve the most relevant context for a user's question.
   - Retrieval is optimized for speed and accuracy using Groq's processing capabilities.

2. **Generation Phase:**
   - The retrieved context is fed into a generative AI model.
   - The model generates a coherent and contextually accurate response based on the retrieved information.

3. **User Interaction:**
   - Responses are displayed in the Streamlit UI, creating a Q&A experience.

## Getting Started

Follow these steps to set up and run the chatbot locally.

### Prerequisites

- Python 3.10+

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/kapishluhariwala/RAG-Document-Q-A-Groq.git
   cd RAG-Document-Q-A-Groq
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Chatbot

1. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to:

   ```
   http://localhost:8501
   ```

3. Set up your document database:
   - Upload the PDF files and click Create Embeddings .

4. Populate a Groq API Key and HuggingFace token

5. Interact with the chatbot through the user-friendly interface.

---

## File Structure

```plaintext
.
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── pdf/                  # Folder to store document data
```

---

## Contact

For questions or support, reach out to:

- **Kapish Luhariwala:** [kapishluhariwala@hotmail.com](mailto\:kapishluhariwala@hotmail.com)
- **GitHub:** [@kapishluhariwala](https://github.com/kapishluhariwala)


