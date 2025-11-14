import math
from collections import defaultdict
from typing import List, Tuple, Any


class MultinomialNaiveBayes:
    """
    Custom Multinomial Naive Bayes for text classification

    Expects training data in the form:
        data = [
            (["<s>", "i", "feel", "sad", "</s>"], 1),
            (["<s>", "today", "was", "ok", "</s>"], 0),
            ...
        ]
    where the second element is the class label (e.g., 0/1),
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

        self.log_class_priors: dict[Any, float] = {}

        self.token_counts: dict[Any, defaultdict[str, int]] = {}

        self.class_token_totals: dict[Any, int] = {}

        self.vocab: set[str] = set()

        self._class_denominators: dict[Any, float] = {}

        self._fitted: bool = False

    def fit(self, data: List[Tuple[List[str], Any]]) -> None:
        """
        Fits the Multinomial Naive Bayes model on training data.

        Parameters
        ----------
        data: list of (tokens, label)
             tokens: list of strings for one document
             label: class for that document (e.g., 0 or 1)
         """
        if not data:
            raise ValueError("Training data is empty.")

        class_doc_counts: defaultdict[Any, int] = defaultdict(int)

        self.token_counts = {}
        self.class_token_totals = {}
        self.vocab = set()

        for tokens, label in data:
            class_doc_counts[label] += 1

            if label not in self.token_counts:
                self.token_counts[label] = defaultdict(int)
                self.class_token_totals[label] = 0

            for w in tokens:
                self.vocab.add(w)
                self.token_counts[label][w] += 1
                self.class_token_totals[label] += 1

        num_docs = len(data)
        class_priors = {}
        for c, count in class_doc_counts.items():
            prior = count / num_docs
            log_prior = math.log(prior)
            class_priors[c] = log_prior

        self.log_class_priors = class_priors

        V = len(self.vocab)
        class_denominators = {}

        for c in self.class_token_totals:
            total_tokens_in_class = self.class_token_totals[c]
            smoothing_term = self.alpha * V

            denominator = total_tokens_in_class + smoothing_term

            class_denominators[c] = denominator

        self._class_denominators = class_denominators
        self._fitted = True
