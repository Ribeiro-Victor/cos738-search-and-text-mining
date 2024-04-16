import xml.etree.ElementTree as ET

SRC_FOLDER_PATH = ''

class QueryProcessor:

    def __init__(self) -> None:
        self.config = self.read_config_file(SRC_FOLDER_PATH + 'PC.CFG')
        self.data_folder_path = SRC_FOLDER_PATH + 'data/'
        self.result_folder_path = SRC_FOLDER_PATH + 'result/'

    def read_config_file(self, path: str):
        config = {
            'read': None,
            'query': None,
            'expected': None
        }

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

        return config
    
    def read_input_file(self):
        filepath = self.data_folder_path + self.config['read']
        tree = ET.parse(filepath)
        return tree.getroot()

    def write_processed_queries_to_file(self, root_tree: ET.Element):
        query_filepath = self.result_folder_path + self.config['query']
        with open(query_filepath, 'w') as file:
            file.write("QueryNumber;QueryText\n")

            for query in root_tree.findall('QUERY'):
                query_number = int(query.find('QueryNumber').text.strip())
                query_text = query.find('QueryText').text.strip().replace(';', '')
                clean_text = ' '.join(query_text.split())
                file.write(f'{query_number};{clean_text}\n')
    
    def calculate_score(self, score: str):
        #Score in format ABCD
        if len(score)!=4:
            return 0
        return int(score[0]) + int(score[1]) + int(score[2]) + int(score[3])

    def write_expected_results_to_file(self, root_tree: ET.Element):
        expected_filepath = self.result_folder_path + self.config['expected']

        with open(expected_filepath, 'w') as file:
            file.write("QueryNumber;DocNumber;DocVotes\n")

            for query in root_tree.findall('QUERY'):
                query_number = int(query.find('QueryNumber').text.strip())
                for item in query.findall('Records/Item'):
                    doc_number = item.text
                    doc_votes = self.calculate_score(item.get('score'))
                    file.write(f'{query_number};{doc_number};{doc_votes}\n')

    def run(self):
        xml_input = self.read_input_file()
        self.write_processed_queries_to_file(xml_input)
        self.write_expected_results_to_file(xml_input)

qp = QueryProcessor()
qp.run()