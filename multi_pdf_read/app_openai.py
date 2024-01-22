import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import io
import docx

load_dotenv()
open_api_key = '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'
os.environ["OPENAI_API_KEY"]=open_api_key

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
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
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


def user_input(vector_search_text):
    embeddings=OpenAIEmbeddings(model="text-embedding-ada-002")

    new_db = FAISS.load_local("faiss_index", embeddings)
    input_text = new_db.similarity_search(vector_search_text)
    #print(docs)
    return input_text


def openai_api_call(input_text, user_incident):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0.5)

    messages = [
    SystemMessage(content="You are a Assistant to a human technician who is operating a baggage handling system in a huge airport.Your job is to \
                  produce probable contingency steps, root causes of such incidents in the past and predict downtime of a component approximately based on such incidents which occured in the past. As refence, you will be given \
                  a summary of past incidents which happened on the specific component, along with the contigency steps taken, root cause and downtime faced. The current incident also be given \
                  as input by the human operator."),
    HumanMessage(content=f'CONTEXT: {input_text}'),
    HumanMessage(content=f'INCIDENT: {user_incident}'),
    HumanMessage(content='As per the given context, what could be specific contingencies, root cause and downtime? Calculate the downtime based on the timestamps given. Be as specific as possible.'),
    AIMessage(content='CONTINGENCY STEPS: '),
    AIMessage(content='PROBABLE ROOT CAUSE: '),
    AIMessage(content='DOWNTIME: '),
    ]

    chat(messages)
    #print(response)
    st.write("Text to llm: ", input_text)
    st.write("Reply: ", chat(messages).content)


def main():
    st.set_page_config("AI Assistant")
    st.header("AI Assistant")

    vec_search= st.text_input("Vector Search")
    user_incident = st.text_input("Incident")
    user_incident12 = st.text_input("Incident12")
 
    if st.button("Submit"):
        if vec_search:
            input_text = user_input(vec_search)
            if user_incident:
                openai_api_call(input_text, user_incident)

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


# 4650 unloader incidents summary, root cause, contingencies and downtime
# 4650 unloader became unavailable 