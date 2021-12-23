import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

def entropy(group):
    assert len(group) > 0, "group must be non-empty"
    m = np.min(group)
    group += np.abs(m)
    proportions = np.bincount(group) / len(group)
    return -np.sum([p * np.log2(p) for p in proportions if p > 0])

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    """ Decision tree implementation

    Parameters
    ----------
    min_samples_split : int, optional (default=2)
        Minimum number of samples in each node.

    max_depth : int, optional (default=20)
        Maximum depth of the tree.

    n_feats : int, optional (default=None)
        If not given, it will be the number of features of training set when invoking the
        fit function.

        The number of features adopted in each decision-making.

    Attributes
    ----------
    root : obj (class Node)
        The root node of this tree, where each leaf node carries a decision value.

    """
    def __init__(self, min_samples_split=2, max_depth=20, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
    
    def fit(self, X_train, y_train):
        X_train, y_train = X_train.reshape([-1, X_train.shape[-1]]), y_train.reshape([-1])
        n_features = X_train.shape[-1]

        # grow tree
        self.n_feats = n_features if not self.n_feats else min(self.n_feats, n_features)
        self.root = self._grow_tree(X_train, y_train)
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth == self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedy search
        best_feat, best_thresh = self._best_criterion(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criterion(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # parent E
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # weighted avg child E
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # return _information_gain
        ig = parent_entropy - child_entropy
        return ig


    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
        

    def _most_common_label(self, group):
        counter = Counter(group)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X_test):
        X_test = X_test.reshape([-1, X_test.shape[-1]])
        
        # traverse tree 
        return np.array([self._traverse_tree(x, self.root) for x in X_test])

    def _traverse_tree(self, X, node):
        if node.is_leaf_node():
            return node.value

        if X[node.feature] <= node.threshold:
            return self._traverse_tree(X, node.left)

        return self._traverse_tree(X, node.right)
        
    def evaluate(self, X_test, y_test):
        X_test = X_test.reshape([-1, X_test.shape[-1]])
        y_test = y_test.reshape([-1])

        return np.sum(self.predict(X_test) == y_test) / y_test.size

if __name__ == '__main__':
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    clf = DecisionTree(min_samples_split=4, max_depth=80)
    clf.fit(X_train, y_train)
    acc = clf.evaluate(X_test, y_test)
    print(f"Accuracy = {acc * 100:.1f} %")
