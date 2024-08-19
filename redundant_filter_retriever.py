from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever


class RedundantFilterRetriever(BaseRetriever):
    def get_relevant_documents(self, query):
        return []

    async def aget_relevant_documents(self):
        return []
