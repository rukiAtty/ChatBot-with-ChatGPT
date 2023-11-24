from flask import Flask, render_template, request, jsonify
import os
import pickle
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI

app = Flask(__name__)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "OpenAI API key"

# Load your pickled data (assuming it contains your vector data)
with open("src/models/faiss_store_openai.pkl", "rb") as f:
    VectorStore = pickle.load(f)

# Initialize your chatbot model
llm = ChatOpenAI(model_name='gpt-3.5-turbo')

# Initialize the chain
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])

def ask_question():
    user_question = request.form.get('question')

    # Use your chatbot code to generate a response
    response = chain({"question": user_question}, return_only_outputs=True)
    
    return jsonify(response)
    

    

   

if __name__ == '__main__':
    app.run(debug=True)


