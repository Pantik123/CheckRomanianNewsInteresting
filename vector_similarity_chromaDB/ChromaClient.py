import chromadb

class ChromaClient:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./news_database")
        self.collection = self.client.get_or_create_collection(
            name="news_romanian",
            metadata={"hnsw:space": "cosine"}
        )

    def add_to_collection(self, embedding, metadata, ids_batch):
        self.collection.upsert(
            embeddings=embedding,
            metadatas=metadata,
            ids=ids_batch
        )
    
    def query_collection(self, query_embeddings, n_results=1):
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )


