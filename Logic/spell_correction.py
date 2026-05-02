import pickle
import os

class SpellCorrection:
    def __init__(self, all_documents=None, load_path=None, save_path=None):
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
        if load_path and os.path.exists(load_path):
            self.load(load_path)
        elif all_documents is not None:
            self.all_k_gram_words, self.word_counter = self.k_gramming_and_counting(all_documents)
            if save_path:
                self.save(save_path)
        else:
            self.all_k_gram_words = {}
            self.word_counter = {}

    def k_gram_word(self, word, k=2):
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
        k_grams = set()
        for idx in range(len(word) - k + 1):
            k_grams.add(word[idx:idx + k])

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
        nearest_word_scores = {}

        word_k_grams = self.k_gram_word(word)

        for other, other_k_grams in self.all_k_gram_words.items():
            if other == word:
                continue

            score = self.jaccard_score(word_k_grams, other_k_grams)

            # TODO: shouldn't be hardcoded!
            if score >= 0.8: 
                nearest_word_scores[other] = score

        nearest_words = sorted(
            nearest_word_scores,
            key=nearest_word_scores.get,
            reverse=True
        )[:5]

        return nearest_words
    
    def is_misspelled_word(self, word):
        """
        Check the term frequency of the word 
        and decide weather the word is misspelled or not

        Parameters
        ----------
        word : str
        
        Returns
        -------
        bool 
            True: misspelled
            False: correct
        """
        word = word.lower()
        word_tf = self.word_counter.get(word, 0)
        return word_tf < 5 # TODO: shouldn't be hardcoded!

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

        for word in query.lower().split():

            if self.is_misspelled_word(word):

                candidates = self.find_nearest_words(word)

                if candidates:
                    corrected_query_words.append(candidates[0])
                else:
                    corrected_query_words.append(word)

            else:
                corrected_query_words.append(word)

        return ' '.join(corrected_query_words)