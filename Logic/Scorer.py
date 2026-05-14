import math
from Logic.preprocess import Preprocessor
from collections import Counter


class Scorer:
    def __init__(self, index, number_of_documents, global_df=None):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The inverted index with structure {term: {document_id: tf}}.
        number_of_documents : int
            The number of documents in the collection.
        """
        self.index = index
        self.idf = {}
        self.N = max(int(number_of_documents), 1)
        self._collection_frequencies = None
        self._collection_length = None
        self.global_df = global_df

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.
        """
        query = self._normalize_query(query)

        docs = set()

        for term in query:
            postings = self.index.get(term, {})
            docs.update(postings.keys())

        return list(docs)

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.
        """
        if term in self.idf:
            return self.idf[term]
        
        if self.global_df is not None:
            df = self.global_df.get(term, 0)
        else:
            df = len(self.index.get(term, {}))

        if df == 0:
            val = 0.0
        else:
            val = math.log10(self.N / df)

        self.idf[term] = val
        return val

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.
        """
        return Counter(query)

    def compute_scores_with_vector_space_model(self, query, method):
        """
        Compute scores with vector space model.
        """
        query = self._normalize_query(query)

        parts = method.split(".")
        if len(parts) != 2 or len(parts[0]) != 3 or len(parts[1]) != 3:
            raise ValueError(f"Invalid SMART notation: {method}")

        doc_method = parts[0]
        query_method = parts[1]

        query_tfs = self.get_query_tfs(query)
        docs = self.get_list_of_documents(query)

        scores = {}

        for doc_id in docs: 
            score = self.get_vector_space_model_score(
                query,
                query_tfs,
                doc_id,
                doc_method,
                query_method,
            )
            scores[doc_id] = score

        return scores

    def get_vector_space_model_score(
        self, query, query_tfs, document_id, document_method, query_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.
        """
        terms = sorted(set(query))

        doc_weights = []
        query_weights = []

        for term in terms:
            tf_d = self.index.get(term, {}).get(document_id, 0)
            w_d = self._apply_tf(tf_d, document_method[0])

            if document_method[1] == "t":
                w_d *= self.get_idf(term)
            
            doc_weights.append(w_d)

            tf_q = query_tfs.get(term, 0)
            w_q = self._apply_tf(tf_q, query_method[0])
            
            if (query_method[1] == "t"):
                w_q *= self.get_idf(term)

            query_weights.append(w_q)

        if document_method[2] == "c":
            doc_weights = self._cosine_normalize(doc_weights)
        if query_method[2] == "c":
            query_weights = self._cosine_normalize(query_weights)
        
        score = sum(d * q for d, q in zip(doc_weights, query_weights))

        return score

    def compute_scores_with_okapi_bm25(
        self, query, average_document_field_length, document_lengths
    ):
        """
        Compute scores with Okapi BM25.
        """
        query = self._normalize_query(query)

        docs = self.get_list_of_documents(query)

        scores = {}

        for doc_id in docs: 
            scores[doc_id] = self.get_okapi_bm25_score(
                query,
                doc_id,
                average_document_field_length,
                document_lengths
            )

        return scores

    def get_okapi_bm25_score(
        self, query, document_id, average_document_field_length, document_lengths
    ):
        """
        Returns the Okapi BM25 score of a document for a query.
        """
        k1 = 1.2
        b = 0.75

        dl = document_lengths.get(document_id, 0)
        avgdl = max(average_document_field_length, 1)

        score = 0.0

        query_tfs = Counter(query)

        for term, _ in query_tfs.items():
            tf = self.index.get(term, {}).get(document_id, 0)

            if tf == 0:
                continue

            if self.global_df is not None:
                df = self.global_df.get(term, 0)
            else:
                df = len(self.index.get(term, {}))

            idf = self._bm25_idf(df)

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (dl / avgdl))

            score += idf * (numerator / denominator)

        return score

    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates scores for each document based on the unigram model.
        """
        query = self._normalize_query(query)

        docs = self.get_list_of_documents(query)
        
        scores = {}

        for doc_id in docs:
            scores[doc_id] = self.compute_score_with_unigram_model(
                query,
                doc_id,
                smoothing_method,
                document_lengths,
                alpha,
                lamda
            )

        return scores

    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the unigram score of a document for a query.
        """
        
        self._prepare_collection_stats()

        score = 0.0

        dl = max(document_lengths.get(document_id, 0), 1)

        for term in query:
            tf = self.index.get(term, {}).get(document_id, 0)

            cf = self._collection_frequencies.get(term, 0)

            if self._collection_length > 0:
                p_collection = cf / self._collection_length
            else:
                p_collection = 0.0

            if smoothing_method is None:
                prob = tf / dl if tf > 0 else 1e-12

            elif smoothing_method == "bayes":
                prob = (tf + alpha * p_collection) / (dl + alpha)

            elif smoothing_method == "mixture":
                prob = lamda * (tf / dl) + (1 - lamda) * p_collection

            else:
                prob = tf / dl if tf > 0 else 1e-12

            prob = max(prob, 1e-12)

            score += math.log(prob)

        return score

    def _apply_tf(self, tf, mode):
        """
        Apply term frequency (tf) weighting based on the specified mode.
        mode (str): Weighting scheme:
            - 'n'
            - 'l'

        """
        if tf <= 0:
            return 0.0
        
        if mode == "n":
            return float(tf)
        
        elif mode == "l":
            return 1.0 + math.log10(tf)
        
        raise ValueError(f"Unsupported tf mode: {mode}")

    def _cosine_normalize(self, weights):
        """
        Normalize a vector of term weights using cosine normalization.
        """
        norm = math.sqrt(sum(w * w for w in weights))
        
        if norm == 0:
            return weights
        
        return [w / norm for w in weights]

    def _prepare_collection_stats(self):
        """
        Compute and cache collection-wide statistics for the index.
        """
        if self._collection_frequencies is not None: 
            return 
        
        cf = {} 
        total = 0

        for term, postings in self.index.items():
            term_total = sum(postings.values())

            cf[term] = term_total 
            total += term_total

        self._collection_frequencies = cf
        self._collection_length = total 

    def _normalize_query(self, query):
        if isinstance(query, str):
            preprocessor = Preprocessor()
            query = preprocessor.preprocess_text(query).split()

        return query

    def _bm25_idf(self, df):
        numerator = self.N - df + 0.5
        denominator = df + 0.5

        if numerator <= 0:
            return 0.0

        return math.log(1 + (numerator / denominator))