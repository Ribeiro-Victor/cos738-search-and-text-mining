import json

SRC_FOLDER_PATH = ''

class SearchEngine:

    def __init__(self) -> None:
        self.config = self.read_config_file(SRC_FOLDER_PATH + 'BUSCA.CFG')
        self.data_folder_path = SRC_FOLDER_PATH + 'data/'
        self.result_folder_path = SRC_FOLDER_PATH + 'result/'

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

    def read_vector_model(self):
        filepath = self.result_folder_path + self.config['model']

        vector_model = {}
        with open(filepath, 'r') as f:
            vector_model = json.load(f)

        return vector_model

    def run(self):
        self.read_vector_model()

se = SearchEngine()
se.run()