from sklearn.model_selection import BaseCrossValidator
import numpy as np

class SubjectSplit(BaseCrossValidator):
    
    def __init__(self, dataset):
        """
        Custom splitter for dataset grouped by subjects.
        Args:
            nb_tot (int): Total number of subjects.
        """
        self.nb_tot = len(dataset)
        self.indexes = []
        
        base = 0
        
        for i in range(self.nb_tot):
            length = len(dataset[i])
            self.indexes.append(np.arange(base,base+length))
            base += length

    def split(self, X=None, y=None, groups=None):
        """
        Generate train-validation splits.
        Yields:
            train_idx, val_idx: Indices for training and validation.
        """
        for validation_subject in range(self.nb_tot):
            val_idx = self.indexes[validation_subject]
            
            train_idx = np.concatenate(
                [self.indexes[i] for i in range(self.nb_tot) if i != validation_subject]
            )
            
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splits (equal to the number of subjects).
        """
        return self.nb_tot