import json 
import time
import os

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments



# Словник для мапінгу категорій у числа
label_2_id = {"noise": 0, "contrabanda": 1, "army": 2, "politics": 3}

with open('./data/data_politics.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

formatted_data = {
    "text": [f"{item['title']}. {item['text']}" for item in raw_data],
    "label": [label_2_id[item['label']] for item in raw_data]
}

dataset = Dataset.from_dict(formatted_data)
num_samples = len(dataset)

# Завантажуємо модель
model = SetFitModel.from_pretrained(
    "my_news_model_v1" if os.path.exists("my_news_model_v1") else "paraphrase-multilingual-MiniLM-L12-v2",
    labels=["noise", "contrabanda", "army"]
)

args = TrainingArguments(
    batch_size=8,
    num_epochs=1
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

# --- Навчання з заміром часу ---
print(f"🚀 Навчання почалося на {num_samples} прикладах...")
start_time = time.time()

trainer.train()

end_time = time.time()
# --- Розрахунок метрик ---
total_time_sec = end_time - start_time
time_per_text_ms = (total_time_sec / num_samples) * 1000
texts_per_sec = num_samples / total_time_sec

print("-" * 30)
print(f"✅ Навчання завершено!")
print(f"Загальний час: {total_time_sec:.2f} сек")
print(f"Час: {time_per_text_ms:.2f} мс на один текст")
print(f"Швидкість: {texts_per_sec:.1f} текстів/сек")
print("-" * 30)

# Зберігаємо результат
model.save_pretrained("my_news_model_v")
print("✅ Модель збережена як 'my_news_model_v2'")