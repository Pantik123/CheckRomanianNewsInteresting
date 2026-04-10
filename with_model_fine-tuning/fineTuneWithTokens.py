import json
import time
from Chuncker import Chuncker

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

# Словник для мапінгу категорій у числа
label_2_id = {"noise": 0, "contrabanda": 1, "army": 2, "politics": 3}

with open('./data/dataset.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)


formatted_data = {
    "text": [],
    "label": []
}

chuncker = Chuncker()

for item in raw_data:

    text = f"{item['title']}. {item['text']}"
    chuncks = chuncker.get_chuncks(text)

    for chunck in chuncks:
        normalized = chuncker.normalize_chunck(chunck)
        label = label_2_id[item['label']]

        formatted_data["text"].append(normalized)
        formatted_data["label"].append(label)

print(f"Загальна кількість текстів після чанкінгу: {len(formatted_data['text'])}")

dataset = Dataset.from_dict(formatted_data)
num_samples = len(dataset)


model = SetFitModel.from_pretrained(
    "paraphrase-multilingual-MiniLM-L12-v2",
    labels=["noise", "contrabanda", "army", "politics"]
)

args = TrainingArguments(
    batch_size=16,
    num_epochs=1,
    num_iterations=20
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
model.save_pretrained("my_news_model_v2")
print("✅ Модель збережена як 'my_news_model_v2'")