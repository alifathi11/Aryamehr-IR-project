import numpy as np
import itertools
import random
import hashlib

class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()
        
        words = document.split()
        words_count = len(words)

        for i in range(words_count - k + 1):
            shingle = tuple(words[i:i + k])
            shingles.add(shingle)

        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        doc_shingles = [self.shingle_document(doc) for doc in self.documents]
        
        all_shingles = sorted(set().union(*doc_shingles))
        self.all_shingles = all_shingles

        rows = len(all_shingles)
        cols = len(self.documents)

        characteristics_matrix = np.zeros((rows, cols), dtype=int)

        shingle_to_row_map = {sh: i for i, sh in enumerate(all_shingles)}

        for col, shingles in enumerate(doc_shingles):
            for sh in shingles:
                row = shingle_to_row_map[sh]
                characteristics_matrix[row, col] = 1

        return characteristics_matrix
    
    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        characteristic_matrix = self.build_characteristic_matrix()

        num_rows, num_docs = characteristic_matrix.shape

        signature = np.full((self.num_hashes, num_docs), np.inf)

        for h in range(self.num_hashes):
            perm = np.random.permutation(num_rows)

            for doc in range(num_docs):
                for row_idx in perm:
                    if characteristic_matrix[row_idx, doc] == 1:
                        signature[h, doc] = row_idx
                        break

        return signature.astype(int)

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        num_hashes, num_docs = signature.shape

        if bands * rows_per_band > num_hashes:
            raise ValueError("bands * rows_per_band is greater than signature rows")
    
        buckets = {}

        for band in range(bands):
            start = band * rows_per_band
            end = start + rows_per_band

            for doc_id in range(num_docs):
                band_slice = tuple(signature[start:end, doc_id])
                bucket_id = (
                    band,
                    int(hashlib.md5(str(band_slice).encode()).hexdigest(), 16)
                )

                if bucket_id not in buckets:
                    buckets[bucket_id] = []
                
                buckets[bucket_id].append(doc_id)
        
        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        num_bands = 25
        signature = self.min_hash_signature()
        ans = self.lsh_buckets(signature, num_bands, self.num_hashes//num_bands)
        return ans

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        union = first_set | second_set

        if len(union) == 0: 
            return 0.0
    
        intersection = first_set & second_set

        return len(intersection) / len(union)

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)

def main():
    
    documents = [
        "the cat sat on the mat",
        "the cat sat on mat",
        "machine learning is fun",
        "machine learning is very fun",
        "the dog barked loudly",
        "the cat sat on the mat"
    ]

    lsh = MinHashLSH(documents, num_hashes=100)

    buckets = lsh.perform_lsh()

    print("Buckets with collisions:")
    for k, v in buckets.items():
        if len(set(v)) > 1:
            print(k, v)

    lsh.jaccard_similarity_test(buckets, documents)


if __name__ == '__main__':
    main()