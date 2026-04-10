import spacy

class Chuncker:
    def __init__(self, model_path=None):
        if model_path:
            self.nlp = spacy.load(model_path)
        else:
            self.nlp = spacy.load("ro_core_news_sm")

    def get_chuncks(self, text):
        doc = self.nlp(text)
        paragraphs = text.split('\n')
        final_chuncks = []

        for para in paragraphs:
            para = para.strip()
            if not para: continue

            # Робимо мікро-док для абзацу, щоб порахувати кількість речень
            para_doc = self.nlp(para)
            sentences = [sent.text for sent in para_doc.sents]

            if len(sentences) <= 5:
                final_chuncks.append(para)
            else:
                # Розбиваємо на групи по 3 речення
                for i in range(0, len(sentences), 3):
                    chunk = " ".join(sentences[i : i + 3])
                    final_chuncks.append(chunk)  

        return final_chuncks
    
    def normalize_chunck(self, chunck_text):
        doc = self.nlp(chunck_text)

        formatted_sentences = []

        # Проходимо по кожному реченню в блоці
        for sent in doc.sents:
            # Витягуємо леми (очищені від стоп-слів та пунктуації)
            lemmas = [t.lemma_.lower() for t in sent 
                      if not t.is_stop and t.is_alpha and len(t.text) > 2]

            # Витягуємо чанки (Іменник + Прикметник) саме для цього речення
            pairs = []
            for token in sent:
                if token.pos_ in ["NOUN", "PROPN"]:
                    # Шукаємо прикметники, що залежать від іменника
                    adjs = [child.text.lower() for child in token.children if child.pos_ == "ADJ"]
                    if adjs:
                        pairs.append(f"{token.text.lower()} {' '.join(adjs)}")

            sent_enhanced = " ".join(lemmas + pairs)
            formatted_sentences.append(sent_enhanced)

        # Склеюємо всі оброблені речення назад у блок
        return " ".join(formatted_sentences)