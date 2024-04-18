import pandas as pd
import numpy as np
from numpy.linalg import norm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
SRC_FOLDER_PATH = r'D:\UFRJ\9º Período\Busca e Mineração de Texto\cos738-search-and-text-mining\task_01\src\\'

class SearchEngine:

    def __init__(self) -> None:
        self.config = self.read_config_file(SRC_FOLDER_PATH + 'BUSCA.CFG')
        self.data_folder_path = SRC_FOLDER_PATH + 'data/'
        self.result_folder_path = SRC_FOLDER_PATH + 'result/'
        self.stop_words = set(s.upper() for s in stopwords.words('english')).union(".", ",", ";", "!", "?", ";", "=", "<", "+", "``", "%", "[", "]", "(", ")", '"', "'", ":", "-", "")
        

    def read_config_file(self, path: str):
        config = {
            'model': None,
            'query': None,
            'results': None
        }

        with open(path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                key, value = line.strip().split('=')
                if key == 'MODELO':
                    config['model'] = (value)
                elif key == 'CONSULTAS':
                    config['query'] = value
                elif key == 'RESULTADOS':
                    config['results'] = value

        return config

    def read_vector_model(self) -> pd.DataFrame:
        filepath = self.result_folder_path + self.config['model']
        vector_model = pd.read_csv(filepath, sep=';', index_col=0)
        return vector_model
    
    def read_queries_file(self) -> dict:
        filepath = self.result_folder_path + self.config['query']
        
        query_dict = {}
        with open(filepath, 'r') as f:
            next(f) #Skips header
            lines = f.readlines()
            for line in lines:
                query_num, query_text = line.strip().split(';')
                query_dict[query_num] = query_text
        
        return query_dict

    def tokenize_query_text(self, text: str):
        text = text.replace('/', ' ')
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if (word not in self.stop_words and not word.isdigit())]
        return filtered_tokens
    
    def calculate_query_tf_idf(self, vector_model_df: pd.DataFrame, query_tokens: list[str]):
        query_idf = vector_model_df['idf'].where(vector_model_df.index.isin(query_tokens), 0)
        return query_idf
    
    def calculate_query_similarity(self, vector_model_df: pd.DataFrame, query_tf_idf: pd.Series) -> pd.DataFrame:
        docs_tf_idf_vectors = vector_model_df.values.T
        cosine_similarity = np.dot(docs_tf_idf_vectors, query_tf_idf)/(norm(docs_tf_idf_vectors, axis=1)*norm(query_tf_idf))
        similarity_df = pd.DataFrame(cosine_similarity, index=vector_model_df.columns, columns=['Similarity'])
        similarity_df.drop(similarity_df[similarity_df['Similarity'] == 0].index, inplace=True)
        similarity_df.drop('idf')
        return similarity_df

    def write_output_to_file(self, queries_similarity_df_list: dict):
        filepath = self.result_folder_path + self.config['results']
        
        with open(filepath, 'w') as f:
            f.write(f'QueryNumber;DocsRankInfo\n')
            for query_number, similarity_df in queries_similarity_df_list.items():
                similarity_df.sort_values(by='Similarity', ascending=False, inplace=True)
                rank = 1
                for row in similarity_df.iterrows():
                    l = [rank, row[0], row[1]['Similarity']]
                    rank += 1
                    f.write(f'{query_number};{str(l)}\n')

    def run(self):
        vector_model = self.read_vector_model()
        query_dict = self.read_queries_file()
        query_similarity_result_dict = {}
        for query_number, query_text in query_dict.items():
            query_tokens = self.tokenize_query_text(query_text)
            query_tf_idf = self.calculate_query_tf_idf(vector_model, query_tokens)
            query_similarity_df = self.calculate_query_similarity(vector_model, query_tf_idf)
            query_similarity_result_dict[query_number] = query_similarity_df
        self.write_output_to_file(query_similarity_result_dict)

se = SearchEngine()
se.run()