import re
import string
import json
import csv



class Preprocessor:
    def __init__(self, custom_stopwords_path='./stopwords.txt'):
        """
        Initialize the preprocessor, compile patterns, load components, etc.
        """
        pattern = r'\S*http\S*|\S*www\S*|\S+\.ir\S*|\S+\.com\S*|\S+\.org\S*|\S*@\S*'
        self.url_pattern = re.compile(pattern=pattern, flags=re.IGNORECASE)

        self.punctuation_table = str.maketrans('', '', string.punctuation)

        self.stopwords = set()

        with open(custom_stopwords_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    self.stopwords.add(word)

    def preprocess_text(
            self, 
            text: str, 
            remove_stop_words=True,
            apply_normalization=True
        ) -> str:
        """
        Apply preprocessing pipeline to a single text document.
        """
        if not text: 
            return ""

        text = text.lower()

        text = self.url_pattern.sub(' ', text)

        text = text.translate(self.punctuation_table)

        text = re.sub(r'\d+', ' ', text) # TODO: do we really need to remove digits?!

        text = re.sub(r'\s+', ' ', text).strip()

        tokens = text.split()

        if remove_stop_words:
            tokens = self.remove_stopwords(text)

        if apply_normalization:
            tokens = [self.normalize(token) for token in tokens]

        return ' '.join(tokens)

    def remove_stopwords(self, text: str) -> list:
        """
        Remove stopwords from the text.
        """
        words = text.split()
        return [word for word in words if word not in self.stopwords]
        
    def normalize(self, word: str) -> str:
        """
        Normalize the text by stemming, lemmatization, etc.

        Parameters
        ----------
        word : str
            The word to be normalized.

        Returns
        ----------
        str
            The normalized word.
        """
        # TODO: do real normalization! (lemmatization and stemming)

        suffixes = ['ing', 'edly', 'ed', 'ly', 'es', 's']

        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2: 
                return word[:-len(suffix)]
            
        return word
        

    def preprocess_many(self, documents: list) -> list:
        """
        Apply preprocessing pipeline to a list of documents.
        """
        return [self.preprocess_text(doc) for doc in documents]

def preprocess_docs(docs: list):
    """
    Apply preprocessing to specific fields in a list of documents in-place.
    
    Args:
        docs (list): List of document dictionaries to preprocess
        
    Returns:
        None: Modifies the input list in-place
    
    Notes:
        Preprocesses the following fields: title, description, author
        Handles both string and list field types
    """
    preprocessor = Preprocessor()

    fields = ['title', 'description', 'author']

    for doc in docs:
        for field in fields:
            if field in doc: 
                if isinstance(doc[field], str):
                    doc[field] = preprocessor.preprocess_text(doc[field]).split()
                elif isinstance(doc[field], list):
                    doc[field] = [
                        preprocessor.preprocess_text(item).split()
                        for item in doc[field]
                        if isinstance(item, str)
                    ]

def csv_to_json(csv_file_path, json_file_path):
    """
    Convert a CSV file to JSON format with specific field mapping.
    
    Args:

        csv_file_path (str): Path to the input CSV file
        json_file_path (str): Path where the output JSON file will be saved
        
    Returns:
        None: Writes output directly to JSON file
    
    Notes:
        Maps CSV fields to JSON structure including:
        - id (from bookId)
        - title, author, description
        - genres, characters, languages (split by commas)
        - publish_date, num_pages, avg_rating
    """
    
    docs = []

    with open(csv_file_path, 'r', encoding='utf-8') as csv_file: 
        reader = csv.DictReader(csv_file)

        for row in reader:
            doc = {
                "id": row.get("bookId", ""),
                "title": row.get("title", ""),
                "author": row.get("author", ""),
                "description": row.get("description", ""),
                "genres": [
                    x.strip() for x in row.get("genres", "").split(',')
                    if x.strip()
                ],
                "characters": [
                    x.strip() for x in row.get("characters", "").split(',')
                    if x.strip()
                ],
                "languages": [
                    x.strip() for x in row.get("languages", "").split(',')
                    if x.strip()
                ],
                "publish_date": row.get("publish_date", ""),
                "num_pages": row.get("num_pages", ""),
                "avg_rating": row.get("avg_rating", ""),
            }

            docs.append(doc)
    
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(docs, json_file, indent=4, ensure_ascii=False)

if __name__ == '__main__':

    csv_to_json('top_3000_rated_books.csv', 'crawled.json')
    
    json_file_path = 'crawled.json'
    with open(json_file_path, "r") as file:
        docs = json.load(file)

    preprocess_docs(docs)

    with open('preprocessed.json', "w") as file:
        file.write(json.dumps(docs))
