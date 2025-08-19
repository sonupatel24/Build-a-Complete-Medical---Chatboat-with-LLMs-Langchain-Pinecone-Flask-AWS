from flask import Flask, render_template, request, jsonify
from src.helpers import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompts import *
import os
import traceback
import google.generativeai as genai

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Validate environment variables
if not PINECONE_API_KEY or not GOOGLE_API_KEY:
    print("❌ Error: Missing required environment variables!")
    print(f"PINECONE_API_KEY: {'✓ Set' if PINECONE_API_KEY else '✗ Missing'}")
    print(f"GOOGLE_API_KEY: {'✓ Set' if GOOGLE_API_KEY else '✗ Missing'}")
    print("Please create a .env file with your API keys.")
    exit(1)

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

print("✅ Environment variables loaded successfully")

# Check available Gemini models
try:
    print("🔄 Checking available Gemini models...")
    genai.configure(api_key=GOOGLE_API_KEY)
    models = genai.list_models()
    available_models = [model.name for model in models if 'generateContent' in model.supported_generation_methods]
    print(f"✅ Available models: {available_models}")
except Exception as e:
    print(f"⚠️ Could not check models: {e}")

try:
    # Load embeddings
    print("🔄 Loading embeddings...")
    embeddings = download_hugging_face_embeddings()
    print("✅ Embeddings loaded successfully")

    # Connect to existing Pinecone index
    print("🔄 Connecting to Pinecone...")
    index_name = "medical-chatbot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    print("✅ Connected to Pinecone successfully")

    # Retriever
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("✅ Retriever created successfully")

    # Use Gemini model
    print("🔄 Initializing Gemini model...")
    chatModel = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Use the correct model name
        temperature=0.3,
        max_output_tokens=1024
    )
    print("✅ Gemini model initialized successfully")

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Chains
    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("✅ RAG chain created successfully")

except Exception as e:
    print(f"❌ Error during initialization: {e}")
    print("Traceback:")
    traceback.print_exc()
    exit(1)

# Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        print(f"📝 User message: {msg}")
        
        # Get response from RAG chain
        response = rag_chain.invoke({"input": msg})
        print(f"🤖 RAG response: {response}")
        
        # Extract answer from response
        if isinstance(response, dict) and "answer" in response:
            answer = response["answer"]
        elif isinstance(response, str):
            answer = response
        else:
            answer = str(response)
        
        print(f"💬 Final answer: {answer}")
        return str(answer)
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        print(f"❌ Error in chat route: {e}")
        print("Traceback:")
        traceback.print_exc()
        return error_msg

if __name__ == "__main__":
    print("🚀 Starting Medical Chatbot...")
    print(f"🌐 App will be available at: http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)
