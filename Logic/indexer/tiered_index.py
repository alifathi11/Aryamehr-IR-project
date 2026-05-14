from Logic.indexer.indexes_enum import Indexes, Index_types
from Logic.indexer.index_reader import IndexReader
import json
import os

class TieredIndex:
    def __init__(self, path="indexes/"):
        """
        Initializes the Tiered_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """

        self.index = {
            Indexes.CHARACTERS: IndexReader(path, index_name=Indexes.CHARACTERS).index,
            Indexes.GENRES: IndexReader(path, index_name=Indexes.GENRES).index,
            Indexes.DESCRIPTION: IndexReader(path, index_name=Indexes.DESCRIPTION).index,
        }
        # TODO: should not be hardcoded!
        # feel free to change the thresholds
        self.tiered_index = { 
            Indexes.CHARACTERS: self.convert_to_tiered_index(3, 1, Indexes.CHARACTERS),
            Indexes.DESCRIPTION: self.convert_to_tiered_index(12, 6, Indexes.DESCRIPTION),
            Indexes.GENRES: self.convert_to_tiered_index(4, 2, Indexes.GENRES)
        }
        self.store_tiered_index(path, Indexes.CHARACTERS)
        self.store_tiered_index(path, Indexes.DESCRIPTION)
        self.store_tiered_index(path, Indexes.GENRES)

    def convert_to_tiered_index(
        self, first_tier_threshold: int, second_tier_threshold: int, index_name
    ):
        """
        Convert the current index to a tiered index.

        Parameters
        ----------
        first_tier_threshold : int
            The threshold for the first tier
        second_tier_threshold : int
            The threshold for the second tier
        index_name : Indexes
            The name of the index to read.

        Returns
        -------
        dict
            The tiered index with structure of 
            {
                "first_tier": dict,
                "second_tier": dict,
                "third_tier": dict
            }
        """
        if index_name not in self.index:
            raise ValueError("Invalid index type")

        current_index = self.index[index_name]
        first_tier = {}
        second_tier = {}
        third_tier = {}

        for term, postings in current_index.items():

            for doc_id, tf in postings.items():

                if tf >= first_tier_threshold:
                    first_tier.setdefault(term, {})[doc_id] = tf

                elif tf >= second_tier_threshold:
                    second_tier.setdefault(term, {})[doc_id] = tf

                else: 
                    third_tier.setdefault(term, {})[doc_id] = tf

        return {
            "first_tier": first_tier,
            "second_tier": second_tier,
            "third_tier": third_tier,
        }

    def store_tiered_index(self, path, index_name):
        """
        Stores the tiered index to a file.
        """
        path = os.path.join(path, index_name.value + "_" + Index_types.TIERED.value + "_index.json")
        with open(path, "w") as file:
            json.dump(self.tiered_index[index_name], file, indent=4)


if __name__ == "__main__":
    tiered = TieredIndex(
        path="indexes/"
    )
