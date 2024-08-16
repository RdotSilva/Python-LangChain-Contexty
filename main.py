from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,  # Chunk size is checked first and when we hit chunk size then we look for the next separator
    chunk_overlap=0,
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

# Create a Chroma instance and immediately calculate embeddings for all docs
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb",
)

question = "What is an interesting fact about the English language?"

results = db.similarity_search_with_score(question)

for result in results:
    print("\n")
    print(result.page_content)
