import ast
import re
import pandas as pd
import numpy as np
import logging

logging.basicConfig(format='[%(asctime)s][%(levelname)s - %(className)s] %(message)s', level=logging.INFO, datefmt='%d/%m/%Y - %H:%M:%S')
logger_extra_dict = {'className': 'Indexer'}
logger = logging.getLogger(__name__)

SRC_FOLDER_PATH = r''

class Indexer:
    
    def __init__(self) -> None:
        logger.info('Starting execution...', extra=logger_extra_dict)
        self.config = self.read_config_file(SRC_FOLDER_PATH + 'INDEX.CFG')
        self.result_folder_path = SRC_FOLDER_PATH + 'result/'

    def read_config_file(self, path: str):
        logger.info('Reading config file...', extra=logger_extra_dict)
        config = {
            'read': None,
            'write': None
        }

        with open(path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                key, value = line.strip().split('=')
                if key == 'LEIA':
                    config['read'] = value
                elif key == 'ESCREVA':
                    config['write'] = value

        logger.info('Config file read successfully.', extra=logger_extra_dict)
        return config

    def read_input_file(self):
        logger.info(f'Reading input file: {self.config["read"]}', extra=logger_extra_dict)
        inv_list = {}
        filepath = self.result_folder_path + self.config['read']

        with open(filepath, 'r') as file:
            next(file) # Skips header
            lines = file.readlines()

            for line in lines:
                word, appearence_str = line.strip().split(';')
                appearence_list = ast.literal_eval(appearence_str)
                inv_list[word] = appearence_list
        
        logger.info('Input file read successfully.', extra=logger_extra_dict)
        return inv_list
    
    def write_output_file(self, term_doc_matrix: pd.DataFrame):
        logger.info(f'Writing output files: {self.config["write"]}', extra=logger_extra_dict)
        filepath = self.result_folder_path + self.config['write']
        term_doc_matrix.to_csv(filepath, sep=';')
        logger.info('Output file generated successfully.', extra=logger_extra_dict)

    def create_term_doc_matrix(self, inv_list: dict) -> pd.DataFrame:
        logger.info('Building Document-term matrix...', extra=logger_extra_dict)
        
        regex_pattern = re.compile(r'^[A-Z]{2,}$') #Over 2 characters and only letters
        filtered_inv_list = {}
        for key, value in inv_list.items():
            if regex_pattern.match(key):
                filtered_inv_list[key] = value
        
        terms = filtered_inv_list.keys()
        docs_numbers = list(set().union(*inv_list.values()))

        term_x_doc_df = pd.DataFrame(index=terms, columns=docs_numbers, dtype='float')
        term_x_doc_df.fillna(0, inplace=True)
        
        for term, appearence in filtered_inv_list.items():
            for doc_number in appearence:
                term_x_doc_df.at[term, doc_number] += 1 #Counts the term frequency (tf) in documents
        
        documents_count = len(docs_numbers)

        term_x_doc_df['doc_freq'] = term_x_doc_df.apply(lambda row: (row != 0).sum(), axis=1) #Doc frequency for each term
        term_x_doc_df['idf'] = np.log10(documents_count/term_x_doc_df['doc_freq'])
        term_x_doc_df.drop('doc_freq', axis=1, inplace=True)

        docs_columns = term_x_doc_df.columns[term_x_doc_df.columns != 'idf']  #Exclude 'idf' column
        term_x_doc_df[docs_columns] = term_x_doc_df[docs_columns].mul(term_x_doc_df['idf'], axis=0) #Calculates tf-idf

        logger.info('Document-term matrix built succesfully.', extra=logger_extra_dict)
        return term_x_doc_df

    def run(self):
        inv_list = self.read_input_file()
        term_doc_matrix = self.create_term_doc_matrix(inv_list)
        self.write_output_file(term_doc_matrix)
        logger.info('All processing done. Program exiting.', extra=logger_extra_dict)

if __name__ == '__main__':
    idx = Indexer()
    idx.run()
