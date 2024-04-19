from indexer import Indexer
from query_processor import QueryProcessor
from inverted_index_generator import ReverseListGenerator
from search_engine import SearchEngine
import nltk

if __name__ == '__main__':
    nltk.download('stopwords')
    nltk.download('punkt')
    ReverseListGenerator().run()
    Indexer().run()
    QueryProcessor().run()
    SearchEngine().run()