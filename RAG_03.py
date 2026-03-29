import requests
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

os.environ[
    "OPENAI_API_KEY"] = ""  # Replace with your OpenAI key

raw_data="https://raw.githubusercontent.com/kailashmishra-hub/My-First-RAG/refs/heads/main/data"
response = requests.get(raw_data)
print(response.status_code)
#print(response.text)

rawdata = response.text
with open("mughals.txt","w") as f:
    f.write(rawdata)

loader=TextLoader("mughals.txt")
document=loader.load()
#print(document[0].page_content)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
text_chunks=text_splitter.split_documents(document)
#print(text_chunks)
print("**************************************************************")
#print(text_chunks[1].page_content)


embeddings=OpenAIEmbeddings()
vectorstores=FAISS.from_documents(text_chunks,embeddings)
retriever=vectorstores.as_retriever()
#print(retriever)

template="""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

prompt=ChatPromptTemplate.from_template(template)
llm_model = ChatOpenAI(model_name="gpt-4o-mini")

output_parser=StrOutputParser()

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | output_parser
)

results=rag_chain.invoke("What was the role of cricket in this period?")
print(results)