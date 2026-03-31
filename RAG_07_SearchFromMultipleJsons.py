import os
import json
from langchain_community.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

os.environ[
    "OPENAI_API_KEY"] = ""  # Replace with your OpenAI key


folder_path = "users_data"

documents = []

# Loop through all JSON files
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    with open(file_path) as f:
        data = json.load(f)

        # Case 1: JSON is a list of records
        if isinstance(data, list):
            for record in data:
                text = json.dumps(record, indent=2)
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file}
                    )
                )
        # Case 2: JSON is a single object
        else:
            text = json.dumps(data, indent=2)
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file}
                )
            )

print("Total documents loaded:", len(documents))
print("\nExample document:\n")
print(documents[1].page_content)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":5})


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
    "List down the id whose skill is only Automation"
    "List down the id whose email is Sincere@april.biz"
    "List down the id whose country field has the value as CA"

)

print("\nAnswer:\n")
print(result)