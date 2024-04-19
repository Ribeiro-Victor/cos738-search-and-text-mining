import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging

logging.basicConfig(format='[%(asctime)s][%(levelname)s - %(className)s] %(message)s', level=logging.INFO, datefmt='%d/%m/%Y - %H:%M:%S')
logger_extra_dict = {'className': 'ReverseListGenerator'}
logger = logging.getLogger(__name__)

SRC_FOLDER_PATH = ''

class ReverseListGenerator:

    def __init__(self) -> None:
        logger.info('Starting execution...', extra=logger_extra_dict)
        self.config = self.read_config_file(SRC_FOLDER_PATH + 'GLI.CFG')
        self.data_folder_path = SRC_FOLDER_PATH + 'data/'
        self.result_folder_path = SRC_FOLDER_PATH + 'result/'
        self.stop_words = set(s.upper() for s in stopwords.words('english')).union(".", ",", ";", "!", "?", ";", "=", "<", "+", "``", "%", "[", "]", "(", ")", '"', "'", ":", "-", "")
        self.inverted_list = {}

    def read_config_file(self, path: str):
        logger.info('Reading config file...', extra=logger_extra_dict)
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
        logger.info('Config file read successfully.', extra=logger_extra_dict)
        return config

    def build_single_inverted_index(self, id, text):
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
        logger.info(f'Reading input files: {self.config["read"]}', extra=logger_extra_dict)
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

        logger.info('Input files read successfully.', extra=logger_extra_dict)
        return records_dict
    
    def write_output_file(self, inv_list):
        logger.info(f'Writing output files: {self.config["write"]}', extra=logger_extra_dict)
        filepath = self.result_folder_path + self.config['write']
        with open(filepath, 'w') as f:
            f.write('WORD;APPEARENCE\n')
            for word, count in inv_list.items():
                f.write(word + ';' + str(count) + '\n')
        logger.info('Output file generated successfully.', extra=logger_extra_dict)

    def run(self):
        records_dict = self.read_input_files()
        inv_index = {}
        logger.info('Building inverted index...', extra=logger_extra_dict)
        for id, abstract in records_dict.items():
            single_inv_index = self.build_single_inverted_index(id, abstract)
            for word, count in single_inv_index.items():
                if word in inv_index:
                    inv_index[word] += count
                else:
                    inv_index[word] = count
        logger.info('Inverted index built successfully.', extra=logger_extra_dict)
        self.write_output_file(inv_index)
        logger.info('All processing done. Program exiting.', extra=logger_extra_dict)

if __name__ == '__main__':
    rlg = ReverseListGenerator()
    rlg.run()