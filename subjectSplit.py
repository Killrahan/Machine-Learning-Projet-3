from sklearn.model_selection import BaseCrossValidator
import numpy as np

class SubjectSplit(BaseCrossValidator):
    def __init__(self, nb_tot):
        """
        Custom splitter for dataset grouped by subjects.
        Args:
            nb_tot (int): Total number of subjects.
        """
        self.nb_tot = nb_tot

    def split(self, X, y=None, groups=None):
        """
        Generate train-validation splits.
        Yields:
            train_idx, val_idx: Indices for training and validation.
        """
        subjects = np.arange(self.n_tot)
        subject_indices = [
            np.arange(len(X[i])) + sum(len(X[j]) for j in range(i))
            for i in range(self.n_tot)
        ]

        for val_subject in range(self.n_tot):
            train_subjects = np.delete(subjects, val_subject)
            train_idx = np.concatenate([subject_indices[i] for i in train_subjects])
            val_idx = subject_indices[val_subject]
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splits (equal to the number of subjects).
        """
        return self.nb_tot