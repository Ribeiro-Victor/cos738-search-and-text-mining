import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging
import time

logging.basicConfig(format='[%(asctime)s][%(levelname)s - %(className)s] %(message)s', level=logging.INFO, datefmt='%d/%m/%Y - %H:%M:%S')
logger_extra_dict = {'className': 'ReverseListGenerator'}
logger = logging.getLogger(__name__)

SRC_FOLDER_PATH = ''

class ReverseListGenerator:

    def __init__(self) -> None:
        logger.info('Starting execution...', extra=logger_extra_dict)
        self.config = self.read_config_file(SRC_FOLDER_PATH + 'config/GLI.CFG')
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

        try:
            with open(path, 'r') as file:
                lines = file.readlines()

                for line in lines:
                    key, value = line.strip().split('=')
                    if key == 'LEIA':
                        config['read'].append(value)
                    elif key == 'ESCREVA':
                        config['write'] = value
            logger.info('Config file successfully read.', extra=logger_extra_dict)
        
        except FileNotFoundError:
            logger.error(f'Config file "{path}" not found. Exiting program.', extra=logger_extra_dict)
            exit(1)
        except Exception:
            logger.error(f'Error while reading config file. Please check file structure. Exiting program.', extra=logger_extra_dict)
            exit(1)
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
            try:
                tree = ET.parse(filepath)
                root = tree.getroot()
            except Exception:
                logger.error(f'Input file "{filepath}" not found. Exiting program.', extra=logger_extra_dict)
                exit(1)
            
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

        logger.info('Input files successfully read.', extra=logger_extra_dict)
        logger.info(f'Number of documents read: {len(records_dict)}', extra=logger_extra_dict)
        return records_dict
    
    def write_output_file(self, inv_list):
        logger.info(f'Writing output files: {self.config["write"]}', extra=logger_extra_dict)
        filepath = self.result_folder_path + self.config['write']
        try:
            with open(filepath, 'w') as f:
                f.write('WORD;APPEARENCE\n')
                for word, count in inv_list.items():
                    f.write(word + ';' + str(count) + '\n')
        except Exception:
            logger.error(f'Couldn\'t write output file. Please check folder structure. Exiting program.', extra=logger_extra_dict)
            exit(1)
        logger.info('Output file successfully generated.', extra=logger_extra_dict)

    def run(self):
        records_dict = self.read_input_files()
        inv_index = {}
        
        logger.info('Building inverted index...', extra=logger_extra_dict)
        start_time = time.perf_counter()

        for id, abstract in records_dict.items():
            single_inv_index = self.build_single_inverted_index(id, abstract)
            for word, count in single_inv_index.items():
                if word in inv_index:
                    inv_index[word] += count
                else:
                    inv_index[word] = count

        end_time = time.perf_counter()
        run_time = end_time - start_time
        avg_time = run_time/len(records_dict)
        logger.info('Inverted index successfully built.', extra=logger_extra_dict)
        logger.info(f'Total word processing time for all documents: {run_time: .2e}s', extra=logger_extra_dict)
        logger.info(f'Average processing time per document: {avg_time: .2e}s', extra=logger_extra_dict)

        self.write_output_file(inv_index)
        logger.info('All processing done. Program exiting.', extra=logger_extra_dict)

if __name__ == '__main__':
    rlg = ReverseListGenerator()
    rlg.run()