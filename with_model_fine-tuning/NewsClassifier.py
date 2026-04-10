from setfit import SetFitModel
from Chuncker import Chuncker

chuncker = Chuncker()

class NewsClassifier:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NewsClassifier, cls).__new__(cls)
            model_path = 'my_news_model_v2'
            
            # Завантажуємо модель один раз з правильними лейблами
            cls._instance.model = SetFitModel.from_pretrained(
                model_path,
                labels=["noise", "contrabanda", "army", "politics"]
            )            
        return cls._instance

    def get_interest_score(self, text):
        # Отримуємо ймовірності
        probs = self.model.predict_proba([text])[0]
        
        # Створюємо словник деталей динамічно на основі того, що ВЖЕ є в моделі
        labels = self.model.labels
        details = {labels[i]: round(float(probs[i]), 4) for i in range(len(probs))}

        # Сумуємо все, крім noise
        noise_p = details.get('noise', 0.0)
        interest_score = 1.0 - noise_p

        # Визначаємо головну категорію
        main_label = self.model.predict([text])[0]

        return {
            "score": round(float(interest_score), 4),
            "label": main_label,
            "details": details
        }
    
    def classify_long_text(self, text, chunker=chuncker):
        # 1. Розбиваємо текст на частини
        chunks = chunker.get_chuncks(text)
        if not chunks:
            return None

        # 2. Збираємо результати для кожного чанку
        all_chunks_details = []
        for chunk in chunks:
            res = self.get_interest_score(chunk)
            all_chunks_details.append(res['details'])

        # 3. Агрегація: шукаємо МАКСИМАЛЬНУ ймовірність для кожної категорії
        # (Це краще за середнє значення, бо важлива тема може бути лише в одному абзаці)
        labels = self.model.labels
        final_details = {}

        for label in labels:
            # Витягуємо всі скори для конкретного лейбла з усіх чанків і беремо макс.
            max_val = max([d.get(label, 0.0) for d in all_chunks_details])
            final_details[label] = round(max_val, 4)

        # 4. Розрахунок підсумкового скору
        noise_p = final_details.get('noise', 0.0)
        final_interest_score = 1.0 - noise_p

        # 5. Визначаємо фінальний лейбл (той, де макс. ймовірність серед НЕ noise)
        interest_labels = {k: v for k, v in final_details.items() if k != 'noise'}
        final_label = max(interest_labels, key=interest_labels.get) if interest_labels else "noise"

        # Якщо поріг шуму все одно високий, міняємо лейбл на noise
        if final_details['noise'] > 0.8:
            final_label = "noise"

        return {
            "score": round(final_interest_score, 4),
            "label": final_label,
            "details": final_details
        }