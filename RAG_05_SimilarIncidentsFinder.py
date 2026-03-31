from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.document_loaders import DirectoryLoader

os.environ[
    "OPENAI_API_KEY"] = ""  # Replace with your OpenAI key


loader = DirectoryLoader("Incidents_List", glob="*.txt")
documents = loader.load()
#print(documents)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
text_chunks=text_splitter.split_documents(documents)
# print("Text Chunks are: " , text_chunks)
# print("**************************************************************")


embeddings=OpenAIEmbeddings()
vectorstores=FAISS.from_documents(text_chunks,embeddings)
retriever=vectorstores.as_retriever(search_kwargs={"k":5})
#print(retriever)

template = """
You are an assistant helping to identify similar incidents.
Use the retrieved incident reports to find the similar past incidents.
If you don't find the similar incident , just say that you don't find a similar incident from the past in your current repository

User Incident:
{question}

Similar Incident Details:
{context}

Provide:
1. Similar Incident ID
2. Title
3. Root Cause
4. Resolution
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

results=rag_chain.invoke("Azure pipeline issue ?")
print("******************************************************************************************************************************")
print(results)