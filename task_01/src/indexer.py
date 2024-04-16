import json
import ast
import re
import math

SRC_FOLDER_PATH = ''

class Indexer:
    
    def __init__(self) -> None:
        self.config = self.read_config_file(SRC_FOLDER_PATH + 'INDEX.CFG')
        self.result_folder_path = SRC_FOLDER_PATH + 'result/'

    def read_config_file(self, path: str):
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

        return config

    def read_input_file(self):
        inv_list = {}
        filepath = self.result_folder_path + self.config['read']

        with open(filepath, 'r') as file:
            next(file) # Skips header
            lines = file.readlines()

            for line in lines:
                word, appearence_str = line.strip().split(';')
                appearence_list = ast.literal_eval(appearence_str)
                inv_list[word] = appearence_list
        
        return inv_list
    
    def write_output_file(self, term_doc_matrix):
        filepath = self.result_folder_path + self.config['write']
        with open(filepath, 'w') as f:
            json.dump(term_doc_matrix ,f)

    def create_term_doc_matrix(self, inv_list: dict):
        
        regex_pattern = re.compile(r'^[A-Z]{3,}$') #Over 2 characters and only letters
        filtered_inv_list = {}
        for key, value in inv_list.items():
            if regex_pattern.match(key):
                filtered_inv_list[key] = value

        term_doc_matrix = {}
        for term, appearence in filtered_inv_list.items():
            term_doc_matrix[term] = {}
            for doc_number in appearence:
                term_doc_matrix[term][doc_number] = term_doc_matrix[term].get(doc_number, 0) + 1 # Counts the term frequency in documents
        
        doc_term_counter = {}
        for doc_list in filtered_inv_list.values():
            for doc_number in doc_list:
                doc_term_counter[doc_number] = doc_term_counter.get(doc_number, 0) + 1
        documents_count = len(doc_term_counter)

        idf = {}
        for term in term_doc_matrix.keys():
            idf[term] = math.log(documents_count/len(term_doc_matrix[term]))
        
        for term, appearence in term_doc_matrix.items():
            for doc_number in appearence:
                term_doc_matrix[term][doc_number] *= idf[term] # Calculates tf-idf
        
        return term_doc_matrix

    def run(self):
        inv_list = self.read_input_file()
        term_doc_matrix = self.create_term_doc_matrix(inv_list)
        self.write_output_file(term_doc_matrix)


idx = Indexer()
idx.run()
