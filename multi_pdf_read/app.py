import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import io
#import docx2txt
import docx

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        #print(pdf)
        #pdf = io.BytesIO(pdf)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_doc_text(doc_files):
    text = ""
    for doc_name in doc_files:
        #print(doc_name)
        doc = docx.Document(doc_name)
        #doc = io.BytesIO(doc)
        for para in doc.paragraphs:
            text+= para.text
    #print(text)
    return text


def read_doc(directory):
    file_loader= PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents


def get_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_stores=FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_stores.save_local("faiss_index")


def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not provided in the context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8)

    prompt= PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain=load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    #print(docs)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    
    #print(response)
    st.write("Text to llm: ", docs)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("AI Assistant")
    st.header("AI Assistant")

    vec_search= st.text_input("Vector Search")
    user_question = st.text_input("Component")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(label="Upload your PDF files and click on the Submit",accept_multiple_files=True)
        #print(pdf_docs)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                #raw_text = get_pdf_text(pdf_docs)
                raw_text = get_doc_text(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
    
if __name__== "__main__":
    main()