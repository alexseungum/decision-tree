import numpy as np
class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None, m=0):
        self.max_depth = max_depth
        self.features = feature_labels
        self.m = m # number of attributes (only for bagged attributes)
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        unique, counts = np.unique(y, return_counts=True)
        probs = np.divide(counts, float(len(y)))
        out = -1 * sum([probs[i]*math.log2(probs[i]) for i in range(len(unique))])
        return out

    def information_gain(self, X, y, idx, thresh):
        Hbefore = DecisionTree.entropy(y)
        X0, y0, X1, y1 = self.split(X, y, idx, thresh)
        Hafter = (len(y0) * DecisionTree.entropy(y0) + len(y1) * DecisionTree.entropy(y1))/(len(y0) + len(y1))
        return Hbefore - Hafter, X0, y0, X1, y1
    
    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        l = np.unique(y)
        if len(l) == 1:
            self.pred = l[0]
        elif self.max_depth == 0:
            self.pred = np.argmax(np.bincount(y))
        else:
            # for loop to test all possible splits 
            max_arr = []
            max_gain = 0
            if self.m > 0:
                temp = np.random.randint(0, high=(self.features.size-1), size=self.m)
                X_attbag = X[:, temp]
            else:
                X_attbag = X
            for i in range(X_attbag.shape[1]):
                values = np.unique(X[:, i])
                for j in values:
                    gain, X0, y0, X1, y1 = self.information_gain(X, y, i, j)
                    if gain > max_gain:
                        max_gain = gain
                        max_arr = [i, j, X0, y0, X1, y1]

            if max_gain <= 0 or not max_arr:
                self.pred = np.argmax(np.bincount(y))
            elif (max_arr[2].size) == 0 or (max_arr[4].size) == 0:
                self.pred = np.argmax(np.bincount(y))
            else:
                self.left = DecisionTree(self.max_depth-1, self.features)
                self.right = DecisionTree(self.max_depth-1, self.features)
                self.split_idx = max_arr[0]
                self.thresh = max_arr[1]
                self.left.fit(max_arr[2], max_arr[3])
                self.right.fit(max_arr[4], max_arr[5])

    def predict(self, X):
        return [self.predict_helper(point) for point in X]
    def predict_helper(self, point):
        if self.pred is not None:
            return self.pred
        if point[self.split_idx] >= self.thresh:
            return self.right.predict_helper(point)
        return self.left.predict_helper(point)

    def __repr__(self):
        if self.max_depth == 0:
            return "%s" % (self.pred)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())
