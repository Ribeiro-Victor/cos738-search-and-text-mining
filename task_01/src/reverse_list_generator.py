import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
SRC_FOLDER_PATH = ''

class ReverseListGenerator:

    def __init__(self) -> None:
        self.config = self.read_config_file(SRC_FOLDER_PATH + 'GLI.CFG')
        self.data_folder_path = SRC_FOLDER_PATH + 'data/'
        self.result_folder_path = SRC_FOLDER_PATH + 'result/'
        self.stop_words = set(s.upper() for s in stopwords.words('english')).union(".", ",", ";", "!", "?", ";", "=", "<", "+", "``", "%", "[", "]", "(", ")", '"', "'", ":", "-", "")
        self.inverted_list = {}

    def read_config_file(self, path: str):
        config = {
            'read': [],
            'write': None
        }

        with open(path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                key, value = line.strip().split('=')
                if key == 'LEIA':
                    config['read'].append(value)
                elif key == 'ESCREVA':
                    config['write'] = value

        return config

    def build_single_inverted_list(self, id, text):
        inv_list = {}

        tokens = word_tokenize(text.upper())
        for w in tokens:
            if w not in self.stop_words and not w.isdigit():
                if w not in inv_list.keys():
                    inv_list[w] = [id]
                else:
                    inv_list[w].append(id)
        
        return inv_list

    def read_input_files(self) -> dict:

        records_dict = {}
        
        for filename in self.config['read']:

            filepath = self.data_folder_path + filename
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            for record in root.findall('.//RECORD'):
                record_num = int(record.find('RECORDNUM').text)
                abstract_element = record.find('ABSTRACT')
                if abstract_element != None:
                    abstract = abstract_element.text
                if abstract_element == None:
                    abstract_element = record.find('EXTRACT')
                    if abstract_element != None:
                        abstract = abstract_element.text
                
                records_dict[record_num] = abstract
        
        return records_dict
    
    def write_output_file(self, inv_list):
        filepath = self.result_folder_path + self.config['write']
        with open(filepath, 'w') as f:
            f.write('WORD;APPEARENCE\n')
            for word, count in inv_list.items():
                f.write(word + ';' + str(count) + '\n')

    def run(self):
        records_dict = self.read_input_files()
        inv_list = {}
        for id, abstract in records_dict.items():
            single_inv_list = self.build_single_inverted_list(id, abstract)
            for word, count in single_inv_list.items():
                if word in inv_list:
                    inv_list[word] += count
                else:
                    inv_list[word] = count
        self.write_output_file(inv_list)
        return inv_list

rlg = ReverseListGenerator()
rlg.run()