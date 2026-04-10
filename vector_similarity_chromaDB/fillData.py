import hashlib # для генерації унікальних ID
import json # для зчитування даних

# Імпорт інших модулів
from SentenceModel import SentenceModel
from Chuncker import Chuncker
from ChromaClient import ChromaClient
from SqliteDB import SqliteDB


sql_db = SqliteDB()
chromaClient = ChromaClient()
sentenceModel = SentenceModel()
chunker = Chuncker() 


# Генерує ID на основі тексту заголовка
def generate_id(text) -> str:
    text = text.lower().strip()
    text = text.replace(" ", "_")

    # Генерує стабільний ID на основі тексту заголовка
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# Зберігає в SQLite та ChromaDB
def add_news_to_system(item):
    title = item["title"]
    raw_text = item["text"]
    label = item["label"]
    news_id = generate_id(title)

    # Зберігаємо в SQLite
    sql_db.insert_article(news_id, title, raw_text, label)


    # Готуємо чанки для ChromaDB
    text_to_embed = title+ "\n" + raw_text
    chunks = chunker.get_chuncks(text_to_embed)
    
    # Для запису у ChromaDB
    embeddings_batch = []
    metadatas_batch = []
    ids_batch = []

    for i, chunk_text in enumerate(chunks):
        normalized = chunker.normalize_chunck(chunk_text) # твоя функція
        vector = sentenceModel.get_vector(normalized)
        
        embeddings_batch.append(vector)
        metadatas_batch.append({
            "news_id": news_id,
            "chunk_index": i,
            "label": label
        })
        ids_batch.append(f"{news_id}_{i}")

    # Записуємо в ChromaDB
    if ids_batch:
        chromaClient.add_to_collection(embeddings_batch, metadatas_batch, ids_batch)
    
    print(f"✅ Новину опрацьовано. ID: {news_id} | Чанків: {len(ids_batch)}")




with open('./data/extra_contr.json', 'r', encoding='utf-8') as f:
    news_data = json.load(f)

for item in news_data:
    add_news_to_system(item)


import sqlite3
import chromadb

# Перевірка SQLite
conn = sqlite3.connect("news_db.db")
count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
print(f"Кількість новин у SQLite: {count}")

# Перевірка Chroma
client = chromadb.PersistentClient(path="./news_database")
collection = client.get_collection(name="news_romanian")
print(f"Кількість чанків у ChromaDB: {collection.count()}")