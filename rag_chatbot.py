from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from langchain.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Setup Flask
app = Flask(__name__)
CORS(app)

# Document folder
DOCUMENTS_FOLDER = "./docs"
WEBSITE_LINKS = [
    "https://www.angelone.in/support",  # ✅ Replace with actual links
]

# Load documents
def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".txt") or filename.endswith(".md"):
            docs.extend(TextLoader(path).load())
        elif filename.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
    return docs

# === Load documents from websites ===
def load_website_documents(urls):
    docs = []
    for url in urls:
        try:
            web_docs = WebBaseLoader(url).load()
            docs.extend(web_docs)
        except Exception as e:
            print(f"Failed to load {url}: {e}")
    return docs

# === Load all knowledge sources ===
local_docs = load_documents(DOCUMENTS_FOLDER)
web_docs = load_website_documents(WEBSITE_LINKS)
all_docs = local_docs + web_docs

# Prepare vector database
raw_docs = load_documents(DOCUMENTS_FOLDER)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(all_docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Local LLM (flan-t5-base)
local_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=local_pipeline)

# Build RAG chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Intent handler for greetings & small talk
def handle_small_talk(message: str) -> str:
    message = message.lower().strip()

    small_talk_responses = {
        ("hi", "hello", "hey"): "👋 Hi there! How can I help you today?",
        ("how are you", "how’s it going", "how r u"): "I'm just a bunch of code, but I'm happy to assist you! 😊",
        ("thank you", "thanks", "thx"): "You're welcome! 🙌 Let me know if you need anything else.",
        ("ok", "cool", "great"): "👍 Got it! Let me know if you have more questions.",
        ("bye", "goodbye", "see you"): "Goodbye! 👋 Come back anytime."
    }

    for keywords, reply in small_talk_responses.items():
        if any(kw in message for kw in keywords):
            return reply
    return None

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "Empty message"}), 400
    
    small_talk = handle_small_talk(user_input)
    if small_talk:
        return jsonify({"response": small_talk})
    response = qa_chain.run(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
