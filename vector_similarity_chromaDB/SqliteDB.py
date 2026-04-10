import sqlite3

class SqliteDB:
    def __init__(self, db_path="news_db.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                news_id TEXT PRIMARY KEY,
                title TEXT ,
                full_text TEXT,
                label TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit() 

    def insert_article(self, news_id, title, full_text, label):
        self.cursor.execute('''
            INSERT OR IGNORE INTO articles (news_id, title, full_text, label) 
            VALUES (?, ?, ?, ?)
        ''', (news_id, title, full_text, label))
        self.conn.commit()