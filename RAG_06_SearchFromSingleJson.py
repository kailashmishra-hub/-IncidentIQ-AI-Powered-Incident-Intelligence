import json
import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ[
    "OPENAI_API_KEY"] = ""  # Replace with your OpenAI key

with open("users.json") as f:
    data = json.load(f)

documents = []

for record in data:
    text = json.dumps(record, indent=2)

    documents.append(
        Document(
            page_content=text,
            metadata={"source": "users1.json"}
        )
    )

print("Example document:\n")
print(documents[0].page_content)


embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})


template = """
You are an assistant answering questions about user data.

Use ONLY the provided context.
Do NOT invent information.

Question:
{question}

Context:
{context}

Answer clearly using the context.
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)


result = rag_chain.invoke(
    "List down the names whose country is USA"
    "what is similarity between Leanne Graham and Ervin Howell in terms of their skills"
)

print("\nAnswer:\n")
print(result)