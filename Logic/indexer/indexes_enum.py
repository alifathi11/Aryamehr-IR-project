from enum import Enum


class Indexes(Enum):
    DOCUMENTS = 'documents'
    CHARACTERS = 'characters'
    GENRES = 'genres'
    DESCRIPTION = 'description'

class Index_types(Enum):
    TIERED = 'tiered'
    DOCUMENT_LENGTH = 'document_length'
    METADATA = 'metadata'