import xml.etree.ElementTree as ET
import logging
import time

logging.basicConfig(format='[%(asctime)s][%(levelname)s - %(className)s] %(message)s', level=logging.INFO, datefmt='%d/%m/%Y - %H:%M:%S')
logger_extra_dict = {'className': 'QueryProcessor'}
logger = logging.getLogger(__name__)
SRC_FOLDER_PATH = ''

class QueryProcessor:

    def __init__(self) -> None:
        logger.info('Starting execution...', extra=logger_extra_dict)
        self.config = self.read_config_file(SRC_FOLDER_PATH + 'config/PC.CFG')
        self.data_folder_path = SRC_FOLDER_PATH + 'data/'
        self.result_folder_path = SRC_FOLDER_PATH + 'result/'

    def read_config_file(self, path: str):
        logger.info('Reading config file...', extra=logger_extra_dict)
        config = {
            'read': None,
            'query': None,
            'expected': None
        }

        try:
            with open(path, 'r') as file:
                lines = file.readlines()

                for line in lines:
                    key, value = line.strip().split('=')
                    if key == 'LEIA':
                        config['read'] = value
                    elif key == 'CONSULTAS':
                        config['query'] = value
                    elif key == 'ESPERADOS':
                        config['expected'] = value
        except FileNotFoundError:
            logger.error(f'Config file "{path}" not found. Exiting program.', extra=logger_extra_dict)
            exit(1)
        except Exception:
            logger.error(f'Error while reading config file. Please check file structure. Exiting program.', extra=logger_extra_dict)
            exit(1)

        logger.info('Config file read successfully.', extra=logger_extra_dict)
        return config
    
    def read_input_file(self):
        filepath = self.data_folder_path + self.config['read']
        try:
            tree = ET.parse(filepath)
        except FileNotFoundError:
            logger.error(f'Input file "{self.config["read"]}" not found. Exiting program.', extra=logger_extra_dict)
            exit(1)
        except Exception:
            logger.error(f'Error while reading input file. Please check file structure. Exiting program.', extra=logger_extra_dict)
            exit(1)
        return tree.getroot()

    def write_processed_queries_to_file(self, root_tree: ET.Element):
        logger.info(f'Writing processed queries to file: {self.config["query"]}', extra=logger_extra_dict)
        query_filepath = self.result_folder_path + self.config['query']
        try:
            with open(query_filepath, 'w') as file:
                file.write("QueryNumber;QueryText\n")

                for query in root_tree.findall('QUERY'):
                    query_number = int(query.find('QueryNumber').text.strip())
                    query_text = query.find('QueryText').text.strip().replace(';', '')
                    clean_text = ' '.join(query_text.split())
                    file.write(f'{query_number};{clean_text.upper()}\n')
        except Exception:
            logger.error(f'Couldn\'t write output processed queries file. Please check folder structure. Exiting program.', extra=logger_extra_dict)
            exit(1)
        logger.info('Processed queries file generated successfully.', extra=logger_extra_dict)
    
    def calculate_score(self, score: str):
        #Score in format ABCD
        if len(score)!=4:
            return 0
        return int(score[0]) + int(score[1]) + int(score[2]) + int(score[3])

    def write_expected_results_to_file(self, root_tree: ET.Element):
        logger.info(f'Calculating score and writing expected results to file: {self.config["expected"]}', extra=logger_extra_dict)
        expected_filepath = self.result_folder_path + self.config['expected']

        try:
            with open(expected_filepath, 'w') as file:
                file.write("QueryNumber;DocNumber;DocVotes\n")

                for query in root_tree.findall('QUERY'):
                    query_number = int(query.find('QueryNumber').text.strip())
                    for item in query.findall('Records/Item'):
                        doc_number = item.text
                        doc_votes = self.calculate_score(item.get('score'))
                        file.write(f'{query_number};{doc_number};{doc_votes}\n')
        except Exception:
            logger.error(f'Couldn\'t write output expected results file. Please check folder structure. Exiting program.', extra=logger_extra_dict)
            exit(1)
        logger.info('Expected results file generated successfully.', extra=logger_extra_dict)

    def run(self):
        xml_input = self.read_input_file()
        logger.info(f'Number of queries read: {len(xml_input.findall("QUERY"))}', extra=logger_extra_dict)
        start_time = time.time()
        self.write_processed_queries_to_file(xml_input)
        self.write_expected_results_to_file(xml_input)
        end_time = time.time()
        run_time = end_time - start_time
        avg_time = run_time/len(xml_input.findall("QUERY"))
        logger.info(f'Total processing time for all queries: {run_time: .2e}s', extra=logger_extra_dict)
        logger.info(f'Average processing time per query: {avg_time: .2e}s', extra=logger_extra_dict)
        logger.info('All processing done. Program exiting.', extra=logger_extra_dict)

if __name__ == '__main__':
    qp = QueryProcessor()
    qp.run()