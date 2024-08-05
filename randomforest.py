class RandomForest():

    # feature_labels, max_depth of tree, num trees, num features per bag, num points per bag
    def __init__(self, feature_labels=None, max_depth=8, tree_count=200, m=1, n=200):
        self.features = feature_labels
        self.n = n
        self.m = m
        self.tree_count=tree_count
        self.max_depth = max_depth
        self.decision_trees = [
            DecisionTree(max_depth=self.max_depth, feature_labels=self.features, m=self.m)
            for i in range(self.tree_count)
        ]

    def fit(self, X, y):
        for tree in self.decision_trees:
            data_bag = np.random.randint(0, high=(X.shape[0]-1), size=self.n)
            X_bagged = X[data_bag]
            y_bagged = y[data_bag]
            tree.fit(X_bagged, y_bagged)

    def predict(self, X):
        predictions = []
        for tree in self.decision_trees:
            predictions.append(tree.predict(X))
        predictions = np.array(predictions)
        modes = stats.mode(predictions, axis=0)[0]
        modes = modes.flatten()
        return modes
