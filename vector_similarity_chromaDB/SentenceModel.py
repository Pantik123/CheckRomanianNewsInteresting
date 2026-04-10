import warnings
import os
from sentence_transformers import SentenceTransformer

# Приховуємо попередження
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

class SentenceModel: 
    _instance = None  # Створюємо атрибут класу заздалегідь

    def __new__(cls):
        if cls._instance is None:
            # Створюємо екземпляр
            cls._instance = super(SentenceModel, cls).__new__(cls)

            # Звертаємось саме до cls._instance
            cls._instance.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
        return cls._instance

    # Додаємо __init__, щоб він не перезавантажував модель при кожному виклику ()
    def __init__(self):
        pass

    def get_embedding(self, text):
        return self.embedder.encode(text, convert_to_numpy=True)
    
    def get_vector(self, text):
        return self.embedder.encode(text).tolist()
    
    def encode_batch(self, texts):
        return self.embedder.encode(texts)