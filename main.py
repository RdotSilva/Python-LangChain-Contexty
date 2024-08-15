from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,  # Chunk size is checked first and when we hit chunk size then we look for the next separator
    chunk_overlap=0,
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

for doc in docs:
    print(doc.page_content)
    print("\n")
