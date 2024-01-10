import os
from constants import openai_key
from langchain.llms import openai

import streamlit as st

# streamlit framework
st.title("langchain demo woth OPENAI")
input_text = st.text_input("Search topic")

## OPENAI LLMS
llm = openai(temperature=0.8)

if input_text:
    st.write(llm(input_text))