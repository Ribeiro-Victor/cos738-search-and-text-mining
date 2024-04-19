# COS738 - Busca e Mineração de Texto

## Task 01 - Implementação de um sistema de recuperação em memória segundo o modelo vetorial

- O sistema foi desenvolvido usando um ambiente virtual em python. As bibliotecas necessárias estão descritas no arquivo [Requirements.txt](task_01/src/requirements.txt). 
<br>Para instalar todas de uma vez, basta executar ```pip install -r requirements.txt```.

- O sistema foi divido em quatro módulos separados em arquivos diferentes: [Inverted Index Generator](task_01/src/inverted_index_generator.py), [Indexer](task_01/src/indexer.py), [Query Processor](task_01/src/query_processor.py) e [Search Engine](task_01/src/search_engine.py).

- Cada módulo possui um arquivo de configuração correspondente, localizados na pasta [config](task_01/src/config).

- Os dados de entrada do sistema estão localizados na pasta [data](task_01/src/data).

- Durante a execução do Indexer, será necessário escolher se a medida de term frequency para o modelo vetorial deverá ser normalizada ou não, por meio do terminal.

- Para rodar o programa com os quatro módulos de uma vez, basta executar o comando ``` python main.py``` dentro da pasta [task_01/src](task_01/src).

- Os resultados e arquivos gerados durante o processamento estão localizados na pasta [result](task_01/src/result)
