import pickle
import os

from Logic.preprocess import Preprocessor

class SpellCorrection:
    def __init__(
            self,
            all_documents=None, 
            load_path=None, 
            save_path=None
        ):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str, optional
            The input documents used to build the vocabulary.
        load_path : str, optional
            Path to load precomputed data from.
        save_path : str, optional
            Path to save computed data to.
        """
        self.k = 2
        self.min_freq = 2
        self.similarity_threshold = 0.3
        self.max_candidates = 5

        self.preprocessor = Preprocessor()

        if load_path and os.path.exists(load_path):
            self.load(load_path)
        elif all_documents is not None:
            self.all_k_gram_words, self.word_counter = self.k_gramming_and_counting(all_documents)
            if save_path:
                self.save(save_path)
        else:
            self.all_k_gram_words = {}
            self.word_counter = {}

    def k_gram_word(self, word):
        """
        Convert a word into a set of k-grams.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each k-gram.

        Returns
        -------
        set
            A set of k-grams.
        """
        word = '$' + word + '$'
        
        if len(word) < self.k:
            return {word}
        
        k_grams = set()
        for idx in range(len(word) - self.k + 1):
            k_grams.add(word[idx:idx + self.k])

        return k_grams

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of k-grams.
        second_set : set
            Second set of k-grams.

        Returns
        -------
        float
            Jaccard score.
        """
        intersection = first_set & second_set
        union = first_set | second_set
        
        if not union:
            return 0

        return len(intersection) / len(union)

    def k_gramming_and_counting(self, all_documents):
        """
        k-grams all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_k_gram_words : dict
            A dictionary from words to their k-grams sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_k_gram_words = {}
        word_counter = {}
        
        for doc in all_documents:
            doc = self.preprocessor.preprocess_text(
                doc,
                remove_stop_words=False,
                apply_normalization=False
            )

            words = doc.split()

            for word in words: 

                word_counter[word] = word_counter.get(word, 0) + 1

                if word not in all_k_gram_words:
                    all_k_gram_words[word] = self.k_gram_word(word)

        return all_k_gram_words, word_counter
                
    def save(self, path):
        """
        Save the k-grams data and word counter to a file.
        """
        data = {
            'all_k_gram_words': self.all_k_gram_words,
            'word_counter': self.word_counter
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        """
        Load the shingle data and word counter from a file.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.all_k_gram_words = data['all_k_gram_words']
            self.word_counter = data['word_counter']

    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : str
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        target_k_grams = self.k_gram_word(word)

        scored_candidates = []

        for candidate, candidate_k_grams in self.all_k_gram_words.items():

            if candidate == word:
                continue
            score = self.jaccard_score(target_k_grams, candidate_k_grams)

            if score >= self.similarity_threshold:
                freq = self.word_counter.get(candidate, 0)

                scored_candidates.append((candidate, score, freq))

        scored_candidates.sort(
            key=lambda x: (x[1], x[2]),
            reverse=True
        )        

        return [candidate for candidate, _, _ in scored_candidates[:self.max_candidates]]
    
    def is_misspelled_word(self, word):
        """
        Check the term frequency of the word 
        and decide whether the word is misspelled or not

        Parameters
        ----------
        word : str
        
        Returns
        -------
        bool 
            True: misspelled
            False: correct
        """
        if word not in self.word_counter: 
            return True
        
        word_tf = self.word_counter.get(word, 0)
        return word_tf < self.min_freq
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : str
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        corrected_query_words = []

        query = self.preprocessor.preprocess_text(
            query, 
            remove_stop_words=False,
            apply_normalization=False
        )

        for word in query.split():

            if self.is_misspelled_word(word):

                candidates = self.find_nearest_words(word)

                if candidates:
                    corrected_query_words.append(candidates[0])
                else:
                    corrected_query_words.append(word)

            else:
                corrected_query_words.append(word)

        return ' '.join(corrected_query_words)