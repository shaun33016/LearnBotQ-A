import streamlit as st
import os
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Customize the layout
st.set_page_config(page_title="LearnBot", page_icon="🤖", layout="wide", )
st.markdown(f"""
            <style>
            .stApp {{background-image: url("https://images.unsplash.com/photo-1513185041617-8ab03f83d6c5?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"); 
                     background-attachment: fixed;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)

# function for writing uploaded file in temp
def write_text_file(content, file_path):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
        return False

# set prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Just give the answer, do not add any headings.

{context}

Question: {question}
Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# initialize the LLM & Embeddings
llm = LlamaCpp(model_path=r"C:\Users\shaun\Desktop\DocQA\\llama-7b.ggmlv3.q4_0.bin")
embeddings = LlamaCppEmbeddings(model_path=r"C:\Users\shaun\Desktop\DocQA\\llama-7b.ggmlv3.q4_0.bin")
llm_chain = LLMChain(llm=llm, prompt=prompt)

col1, col2 = st.columns([2, 1])
with col1:
    st.title("Greetings from LearnBot 🤖")
    uploaded_file = st.file_uploader("Upload an article", type="txt")

    if uploaded_file is not None:
        content = uploaded_file.read().decode('utf-8')
        file_path = "temp/file.txt"
        write_text_file(content, file_path)

        loader = TextLoader(file_path)
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)
        db = Chroma.from_documents(texts, embeddings)

        st.success("File Loaded Successfully!!")

        # Query through LLM
        question = st.text_input("What would you like to ask?", placeholder="Insert question here.....", disabled=not uploaded_file,)

        if question:
            similar_doc = db.similarity_search(question, k=1)
            if similar_doc:
                context = similar_doc[0].page_content
                query_llm = LLMChain(llm=llm, prompt=prompt)
                response = query_llm.run({"context": context, "question": question})
                st.write(response)
            else:
                st.warning("No similar document found.")
