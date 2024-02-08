import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from operator import itemgetter
import docx
import re

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
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=20)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_stores=FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_stores.save_local("faiss_index")


def user_input(vector_search_text):
    embeddings=OpenAIEmbeddings(model="text-embedding-ada-002")

    new_db = FAISS.load_local("faiss_index", embeddings)
    input_text = new_db.similarity_search_with_score(vector_search_text, k=6, fetch_k=15)
    final_text.clear()
    st.write(input_text)
    for doc,score in input_text:
        if (score) < 0.5:
            final_text.append(doc)
    return final_text


def openai_api_call(memory, user_incident, retrieval_req_status, components):

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0.5)
    context_template = "context: {data}"
    query_template = "incident: {query}"
    
    chatprompt=ChatPromptTemplate.from_messages([
    ("system","You are a Assistant to a human technician who is operating a baggage handling system in a huge airport.Your job is to \
                  produce probable contingency steps, root causes of such incidents in the past and predict downtime of a component approximately based on such incidents which occured in the past. As refence, you will be given \
                  a summary of past incidents which happened on the specific component, along with the contigency steps taken, root cause and downtime faced. The current incident also be given \
                  as input by the human operator."),
    ("human",context_template),
    ("human",query_template)])
    
    chain2 = (
        {"data": RunnableLambda(user_input),
         "query": RunnablePassthrough()
        } 
        | chatprompt 
        | chat 
        | StrOutputParser())
    
    if retrieval_req_status == True:
        result = chain2.invoke(user_incident)
    else:
        result = chain2.invoke(user_incident + " " + components['Components'])


#    chain2 = (
#        {"data": itemgetter("query") | RunnableLambda(user_input),
#         "query": itemgetter("query")
#        } 
#        | chatprompt 
#        | chat 
#        | StrOutputParser())
    
    #result = chain2.invoke({"query": user_incident, "retrieve_memory": memory})
    
    #memory.save_context({"query":user_incident},{"Context":})
    #st.write("Text to llm: ", memory)
    st.write("Reply: ", result)


def find_subject(query, example_memory, get_components):
    gpt_prompt = PromptTemplate(
        input_variables=["user_query"],
        template = '''Extract the component name from the query as per your knowledge and examples given below, and return a list of component names. If 
        no component is found, then return the string as no components found:
        Examples: 
        {chat_history}
        
        User query: {user_query}
        Answer: '''
    )

    llm = OpenAI(model = "gpt-3.5-turbo-instruct", temperature=0.0)
    ask_gpt_chain = LLMChain(llm=llm, prompt=gpt_prompt, verbose=True, memory=example_memory, output_key="Component")
    return_ans = ask_gpt_chain({"user_query": query})
    #st.write("ans: ", return_ans)
    #st.write("ans: ", return_ans["Component"])
    #st.write("mem: ",example_memory.load_memory_variables({}))

    pattern = re.compile("no components found")
    match = re.search(pattern, return_ans["Component"])
    if match:
        return False
    else:
        get_components.clear()
        get_components['Components'] = return_ans["Component"]
        st.write("components: ", get_components['Components'])
        return True


def initialize_example_mem(ex_mem):
    ex_mem.chat_memory.add_user_message("What is could be the downtime if 5650 unloader has some fault?")
    ex_mem.chat_memory.add_ai_message(["5650 unloader"])
    ex_mem.chat_memory.add_user_message("What is the impact of 7741 Transport line taken out of service?")
    ex_mem.chat_memory.add_ai_message(["7741 Transport line"])
    ex_mem.chat_memory.add_user_message("4650 unloader in terminal 5 became unavailable, what could be the root cause?")
    ex_mem.chat_memory.add_ai_message(["4650 unloader"])
    ex_mem.chat_memory.add_user_message("what could be the downtime?")
    ex_mem.chat_memory.add_ai_message(["no components found"])
    ex_mem.chat_memory.add_user_message("High volumes of product exiting the T3IB bagstore resulted in the failing of the 7741 transport line.What could be the impact overall and downtime?")
    ex_mem.chat_memory.add_ai_message(["T3IB bagstore","7741 Transport line"])
    ex_mem.load_memory_variables({})



def main():
    st.set_page_config("AI Assistant")
    st.header("AI Assistant")

    user_incident = st.text_input("Current Incident")
    
    if st.button("Submit"):
        if user_incident:
            retrieval_req_status = find_subject(user_incident, st.session_state['example_mem'], st.session_state['components'])
            if retrieval_req_status == False and len(st.session_state['components']) == 0:
                st.text("Please mention the component(s).")
            else:
                openai_api_call(st.session_state['incidents'], user_incident, retrieval_req_status, st.session_state['components'])
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
    load_dotenv()
    open_api_key = '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'
    os.environ["OPENAI_API_KEY"]=open_api_key
    final_text = list()
    current_components = dict()
    example_memory = ConversationBufferMemory(memory_key="chat_history")
    initialize_example_mem(example_memory)
    incidents_memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key = "question")
    if 'example_mem' not in st.session_state:
        st.session_state['example_mem'] = example_memory
    if 'components' not in st.session_state:
        st.session_state['components'] = current_components
    if 'incidents' not in st.session_state:
        st.session_state['incidents'] = incidents_memory
    #print(incidents_memory.load_memory_variables({}))
    main()


# 4650 unloader incidents summary, root cause, contingencies and downtime
# 4650 unloader became unavailable
# High volumes of product exiting the T3IB bagstore resulted in the failing of the 7741 transport line.What could be the impact overall and downtime? 
#