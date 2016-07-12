from sklearn.cross_validation import ShuffleSplit
import numpy as np


class LabelShuffleSplit(ShuffleSplit):
    """Shuffle-Labels-Out cross-validation iterator
    Provides randomized train/test indices to split data according to a
    third-party provided label. This label information can be used to encode
    arbitrary domain specific stratifications of the samples as integers.
    For instance the labels could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.
    The difference between LeavePLabelOut and LabelShuffleSplit is that
    the former generates splits using all subsets of size ``p`` unique labels,
    whereas LabelShuffleSplit generates a user-determined number of random
    test splits, each with a user-determined fraction of unique labels.
    For example, a less computationally intensive alternative to
    ``LeavePLabelOut(labels, p=10)`` would be
    ``LabelShuffleSplit(labels, test_size=10, n_iter=100)``.
    Note: The parameters ``test_size`` and ``train_size`` refer to labels, and
    not to samples, as in ShuffleSplit.
    .. versionadded:: 0.17
    Parameters
    ----------
    labels :  array, [n_samples]
        Labels of samples
    n_iter : int (default 5)
        Number of re-shuffling and splitting iterations.
    test_size : float (default 0.2), int, or None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the labels to include in the test split. If
        int, represents the absolute number of test labels. If None,
        the value is automatically set to the complement of the train size.
    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the labels to include in the train split. If
        int, represents the absolute number of train labels. If None,
        the value is automatically set to the complement of the test size.
    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.
    """
    def __init__(self, labels, n_iter=5, test_size=0.2, train_size=None,
                 random_state=None):

        classes, label_indices = np.unique(labels, return_inverse=True)

        super(LabelShuffleSplit, self).__init__(
            len(classes),
            n_iter=n_iter,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state)

        self.labels = labels
        self.classes = classes
        self.label_indices = label_indices

    def __repr__(self):
        return ('%s(labels=%s, n_iter=%d, test_size=%s, '
                'random_state=%s)' % (
                    self.__class__.__name__,
                    self.labels,
                    self.n_iter,
                    str(self.test_size),
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iter

    def _iter_indices(self):
        for label_train, label_test in super(LabelShuffleSplit,
                                             self)._iter_indices():
            # these are the indices of classes in the partition
            # invert them into data indices

            train = np.flatnonzero(np.in1d(self.label_indices, label_train))
            test = np.flatnonzero(np.in1d(self.label_indices, label_test))

            yield train, test

