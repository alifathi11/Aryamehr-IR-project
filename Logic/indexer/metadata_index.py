from Logic.indexer.index_reader import IndexReader
from Logic.indexer.indexes_enum import Indexes, Index_types
import json
import os

class MetadataIndex:
    def __init__(self, path='indexes/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """
        self.documents = self.read_documents(path)
        self.metadata_index = self.create_metadata_index()
        self.store_metadata_index(path)

    def read_documents(self, path):
        """
        Reads the documents.
        """
        return IndexReader(
            path=path,
            index_name=Indexes.DOCUMENTS
        ).index

    def create_metadata_index(self):    
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['average_document_length'] = {
            'characters': self.get_average_document_field_length('characters'),
            'genres': self.get_average_document_field_length('genres'),
            'description': self.get_average_document_field_length('description')
        }
        metadata_index['document_count'] = len(self.documents)

        return metadata_index
    
    def get_average_document_field_length(self, where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """
        
        total_length = 0
        doc_count = len(self.documents)

        for _, doc in self.documents.items():
            field_index = doc.get(where, [])
            total_length += len(field_index)

        if doc_count == 0:
            return 0

        return total_length / doc_count

    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        path = os.path.join(path, Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json')
        with open(path, 'w') as file:
            json.dump(self.metadata_index, file, indent=4)

    
if __name__ == "__main__":
    meta_index = MetadataIndex('indexes/')