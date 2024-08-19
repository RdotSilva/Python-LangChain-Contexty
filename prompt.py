from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings,
)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
)

question = "What is an interesting fact about the English language?"

result = chain.invoke(question)

print(result)
