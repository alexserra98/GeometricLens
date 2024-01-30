import sqlite3

class MetadataDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY,
            id_hd TEXT,
            id_logits TEXT,
            dataset TEXT,
            train_instances INTEGER,
            model_name TEXT,
            loss REAL,
            std_pred TEXT,
            only_ref_pred TEXT,
            letter_gold TEXT,
            method TEXT
        );
        '''
        self.conn.execute(create_table_query)
        self.conn.commit()

    def add_metadata(self, entry):
        if isinstance(entry, list):
            for e in entry:
                self._add_metadata(e)
        else:
            self._add_metadata(entry)
            
    def _add_metadata(self, entry):
        if self.query_metadata(f'id_hd="{entry.id_hd}"'):
            return
        insert_query = '''
        INSERT INTO metadata (id_hd, 
                              id_logits,
                              dataset, 
                              train_instances, 
                              model_name, 
                              loss, 
                              std_pred, 
                              only_ref_pred, 
                              letter_gold, 
                              method)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        self.conn.execute(insert_query, (entry.id_hd, 
                                         entry.id_logits,
                                         entry.dataset, 
                                         entry.train_instances, 
                                         entry.model_name, 
                                         entry.loss, 
                                         entry.std_pred, 
                                         entry.only_ref_pred, 
                                         entry.letter_gold, 
                                         entry.method))
        self.conn.commit()

    def query_metadata(self, condition):
        query = f'SELECT * FROM metadata WHERE {condition};'
        cursor = self.conn.execute(query)
        return cursor.fetchall()

    def close(self):
        self.conn.close()

