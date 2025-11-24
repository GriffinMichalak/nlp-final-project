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

    def _log_prob_doc_given_class(self, tokens: List[str], c: Any) -> float:
        """
        Compute the log probability of a document given a class label,

        Parameters
        ----------
        tokens : list[str]
            The tokenized document
        c : Any
            The class label (e.g., 0 or 1).

        Returns
        -------
        float
            The log-likelihood log P(x | y = c).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before computing likelihoods.")

        log_prob = 0.0
        denom = self._class_denominators[c]
        counts_c = self.token_counts[c]

        for w in tokens:
            count_wc = counts_c.get(w, 0)
            num = count_wc + self.alpha
            log_prob += math.log(num / denom)

        return log_prob

    def predict_one(self, tokens: List[str]) -> Any:
        """
        Predict the class label for a single tokenized document.

        Parameters
        ----------
        tokens: list[str]
            The tokenized document to classify

        Returns
        -------
        Any
            The predicted class label (e.g., 0 or 1).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict_one().")

        best_class = None
        best_score = float("-inf")

        for c, log_prior in self.log_class_priors.items():
            score = log_prior + self._log_prob_doc_given_class(tokens, c)
            if score > best_score:
                best_score = score
                best_class = c

        return best_class

    def predict(self, data: List[List[str]]) -> List[Any]:
        """
        Predict class labels for a list of tokenized documents.

        Parameters
        ----------
        data: list of list[str]
            A list where each element is a tokenized document.

        Returns
        -------
        list[Any]
            A list of predicted class labels.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        predictions = []
        for tokens in data:
            pred = self.predict_one(tokens)
            predictions.append(pred)

        return predictions

    def score(self, data: List[Tuple[List[str], Any]]) -> float:
        """
        Compute accuracy of the model on labeled data.

        Parameters
        ----------
        data : list of (tokens, label)
            Each element is (tokenized_document, true_label).

        Returns
        -------
        float
            Classification accuracy in [0, 1].
        """
        if not data:
            raise ValueError("Data is empty.")

        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling score().")

        correct = 0
        total = 0

        for tokens, true_label in data:
            pred = self.predict_one(tokens)
            if pred == true_label:
                correct += 1
            total += 1

        return correct / total


