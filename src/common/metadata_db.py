import sqlite3
from .utils import retry_on_failure

class MetadataDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY,
            id_instance TEXT,
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

    def add_metadata(self, metadata_list):
        insert_query = '''
        INSERT INTO metadata (id_instance,
                              dataset, 
                              train_instances, 
                              model_name, 
                              loss, 
                              std_pred, 
                              only_ref_pred, 
                              letter_gold, 
                              method)
        VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        
        try:
            counter = 0
            for metadata in metadata_list:
                if self.row_already_exists(metadata):
                    counter +=1
                    metadata_list.remove(metadata)
            print(f'{counter} elements already in the DB\n')
            
            self.conn.executemany(insert_query, metadata_list)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"Exception in insert operation: {e}")
    
    def row_already_exists(self, entry):
        return self.query_metadata(f'id_instance = "{entry.id_instance}"')
            
    def add_single_metadata(self, entry):
        if self.query_metadata(f'id_instance = "{entry.id_instance}"'):
            return
        insert_query = '''
        INSERT INTO metadata (id_instance,
                              dataset, 
                              train_instances, 
                              model_name, 
                              loss, 
                              std_pred, 
                              only_ref_pred, 
                              letter_gold, 
                              method)
        VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        self.conn.execute(insert_query, (entry.id_instance,
                                         entry.dataset, 
                                         entry.train_instances, 
                                         entry.model_name, 
                                         entry.loss, 
                                         entry.std_pred, 
                                         entry.only_ref_pred, 
                                         entry.letter_gold, 
                                         entry.method))
        self.conn.commit()

    @retry_on_failure(3)
    def query_metadata(self, condition):
        query = f'SELECT * FROM metadata WHERE {condition};'
        cursor = self.conn.execute(query)
        return cursor.fetchall()

    def close(self):
        self.conn.close()

