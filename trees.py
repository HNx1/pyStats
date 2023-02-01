# In this file we will build up Bagging, Random Forests and Adaboost with SAMME for decision trees,
# using only numpy for random numbers and simple functions not in pure python like np.exp
# We first define a DTC class that replicates some needed functionality from sklearn.tree.DecisionTreeClassifier

# We expect datasets that look like lists of length n, where n is the number of observations in the dataset.
# The items in this list will be lists/1-D np.ndarrays (e.g. if produced by list comprehension from a pandas dataframe) of length (p+1)
# Each item will be a 2 tuple, the index of the observation (for accurate sample weighting) and a (p+1) length vector
# where the first p elements are the p features of each observation, and the last element is the class.

# For example, [item[1][-1] for item in dataset] returns the list of classifications for the observations in order

# We preserve indices for two reasons: 1) consistency across methods in DTC operation and
# 2) allows flexibility to implement cross-validation/bootstrapping in the dataset prior to boosting, which needs to know the index of the observation to apply the correct weight.

# Note the DTC.split method is O(p*n**2) so for datasets > n=1000,p=100 this gets noticeably slow.
# Consider vectorising DTC.split/DTC.gini using numpy for a faster implementation that is still close to pure python


import numpy as np


class DTC():
    def __init__(self, dataset, min, max, weights=None, rf=False):
        self.weights = weights if weights else np.ones(
            np.max(list(map(lambda x: x[0], dataset)))+1)

        def createTree(dataset, min, max, rf):
            start = self.split(dataset, rf)
            self.treeSplit(start, min, max, 1, rf)
            return start
        self.tree = createTree(dataset, min, max, rf)

    # Helper function for Gini Index based on split of the data, with observation weights
    def gini(self, groups, classes):
        n = sum(self.weights)
        gini = 0.0
        for group in groups:
            if not group:
                continue
            score = 0.0
            t_g = sum(self.weights[i] for i, _ in group)
            for cla in classes:
                p = sum(self.weights[i]
                        for i, item in group if item[-1] == cla)/t_g
                score += p**2
            gini += (1.0-score)*(t_g/n)
        return gini

    # Split dataset based on value of an attribute
    def splitAttr(self, dataset, attr, val):
        x, y = [], []
        for item in dataset:
            if item[1][attr] < val:
                x.append(item)
            else:
                y.append(item)
        return x, y

    # Find the best attribute to split a dataset on and return an object that acts as a node in a decision tree.
    def split(self, dataset, rf):
        # If random forest,randomly pick sqrt(len(p)) features from feature list p for which to consider list
        allowList = np.random.choice(len(dataset[0][1]), round(
            np.sqrt(len(dataset[0][1])-1)), replace=False) if rf else list(range(len(dataset[0][1])-1))
        # Potential outcomes
        outs = list({item[1][-1] for item in dataset})
        giniMin = 10e6
        # Loop over data items and indices to find best gini index
        for item in dataset:
            for i in range(len(dataset[0][1])-1):
                if i in allowList:
                    groups = self.splitAttr(dataset, i, item[1][i])
                    gin = self.gini(groups, outs)
                    if gin < giniMin:
                        giniMin, attr, val, nodes = gin, i, item[1][i], groups
        # Return a dict object: here we keep children so we can move down the tree, the attribute on which this node splits, and the value that determines which child to go to)
        return {"attr": attr, "val": val, "children": nodes}

    # Create end node with classifier based on majority vote. Used when depth/node size constraints are hit.
    def end(self, list):
        outs = [item[1][-1] for item in list]
        return max(set(outs), key=outs.count)

    # Split a tree at a node

    def treeSplit(self, node, min, max, depth, rf):
        # Takes min (size), max (depth), and current depth inputs.
        # Get the groups of data on each branch of the tree at current node
        l, r = node["children"]
        if not l or not r:
            # If either empty, terminate at this node based on what remains in other
            node["l"], node["r"] = self.end(l+r), self.end(l+r)
            return
        elif depth >= max:
            # If too deep, terminate each node in place
            node["l"], node["r"] = self.end(l), self.end(r)
            return
        else:
            # If node too small, end there, otherwise split further
            if len(l) <= min:
                node["l"] = self.end(l)
            else:

                node["l"] = self.split(l, rf)
                self.treeSplit(node["l"], min, max, depth+1, rf)
            if len(r) <= min:
                node["r"] = self.end(r)
            else:
                node["r"] = self.split(r, rf)
                self.treeSplit(node["r"], min, max, depth+1, rf)

    # Use the tree to make a prediction from a node, picks route based on value, then checks if node is an end or not, either retrieving the value or going deeper.
    def nodePredict(self, node, item):
        if item[node["attr"]] < node["val"]:
            return self.nodePredict(node["l"], item) if isinstance(node["l"], dict) else node["l"]
        else:
            return self.nodePredict(node["r"], item) if isinstance(node["r"], dict)else node["r"]

    # Makes predictions from top of tree
    def treePredict(self, item):
        return self.nodePredict(self.tree, item)

    # Make an individual prediction

    def DTCpred(self, trees, item):
        outs = [self.nodePredict(tree, item) for tree in trees]
        return max(set(outs), key=outs.count)

# ******BAGGING******

# Bagging is used to reduce the high variance of decision trees, as different training sets (sampled randomly from within one large training set) can lead to dramatically different trees.
# Basic theoretical idea: given n independent X_i with constant variance s**2, sample once from each and take mean to get sample mean X. This has variance s**2/n

# We use k splitting of the data for cross-validation to establish error terms.
# For computational efficiency, for observation i you could average across all trees where observation i is out of bag to generate a predicted response.
# This requires some slight refactoring of the code to achieve

# We use the gini index to split based on optimal classifier at each node in the tree.

# An optional rf parameter can be supplied into the bagging/scoreBagging functions to implement a random forest, which restricts the feature choice at a split point to one of sqrt(p) features, randomly chosen.
# An optional ratio parameter can be supplied to reduce the size of the sample used in the bagging algorithm (by default it is equal to the number of the observations, so asymptotically containing 1-e**-1 proportion of the observations)


# Make an individual bagging prediction
def baggingPred(trees, item):
    outs = [tree.treePredict(item) for tree in trees]
    return max(set(outs), key=outs.count)


# The bagging algorithm
def bagging(train, test, min, max, n, ratio=1.0, rf=False):
    # Create n trees, predict outcome for each item in the test data using these trees
    trees = [DTC(subsample(train, ratio)[0], min, max, rf=rf)
             for _ in range(n)]
    return [baggingPred(trees, item) for item in test]


# Split the dataset for k-fold cross validation purposes
def k_split(dataset, k):
    dataset_split = []
    dataset_copy = dataset.copy()
    target_size = int(len(dataset)/k)
    for _ in range(k):
        fold = []
        while len(fold) < target_size:
            index = np.random.randint(0, len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Sample with replacement.
def subsample(dataset, ratio=1.0):
    sample, oob, valList = [], [], []
    lenTarget = round(len(dataset)*ratio)
    while len(sample) < lenTarget:
        val = np.random.randint(len(dataset))
        sample.append(dataset[val])
        valList.append(val)
    oob = [dataset[x]
           for x in range(len(dataset)) if x not in set(valList)]
    return sample, oob


# Create n samples from a dataset
def nSamples(dataset, n, ratio=1.0):
    sampleList = []
    while len(sampleList) < n:
        sampleList.append(subsample(dataset, ratio))
    return sampleList


# Helper function that returns percent of a list that matches another list in the same index
def accuracy(list1, list2):
    return 100.0*sum(map(lambda x, y: 1 if x == y else 0, list1, list2))/len(list1)


# Score the bagging algorithm
def scoreBagging(dataset, k_folds, min, max, n_trees, ratio=1.0, rf=False):
    folds = k_split(dataset, k_folds)
    total = 0
    for i in range(k_folds):
        train_set = list(folds)
        fold = train_set.pop(i)
        train_set = sum(train_set, [])
        test_set = [item[1][:-1] for item in fold]
        preds = bagging(train_set, test_set, min, max, n_trees, ratio, rf)
        truth = [item[1][-1]for item in fold]
        total += accuracy(truth, preds)
    return total/k_folds

# ******BOOSTING******

# Boosting is used to build classifiers sequentially, using a weak learner to fit the residuals of the overall,
# then combining this weak learner parametrically into the tree

# SAMME is an extension of AdaBoost to multiple classes (AdaBoost takes only {-1,1} as classes)
# We show prediction functions separately for them but SAMME reduces to AdaBoost in the two dimensional case. You can use AdaBoost.predictSamme all the time.


class AdaBoost():
    def __init__(self, obCount, dataset):
        self.classifiers = []
        self.alphas = []
        self.weights = [1/obCount]*obCount
        self.dataset = dataset
        self.model = None
        self.features = list(set(item[1][-1] for item in dataset))

    def reset(self):
        self.classifiers = []
        self.alphas = []
        self.weights = [1/len(self.weights)]*len(self.weights)
        self.model = None

    def train(self, rounds=50):
        # 1. To allow continuous training, we offload the initialisation step to the __init__ or reset methods
        # 2. For a number of iterations, do subsequent steps
        for _ in range(rounds):
            # (a) Fit a classifier using the weights
            dtc = DTC(self.dataset, min=2, max=2, weights=self.weights)
            truths = [item[1][-1] for item in self.dataset]
            preds = [dtc.treePredict(item[1][:-1])
                     for item in self.dataset]
            # (b) Compute the error
            error = sum(self.weights[i]*(truths[i] != preds[i])
                        for i in range(len(truths)))/sum(self.weights)
            # (c) Compute Alpha (for SAMME, but if only two features reduces to Adaboost)
            alpha = np.log((1-error)/error) + np.log(len(self.features)-1)
            self.alphas.append(alpha)
            self.classifiers.append(dtc)
            # (d) Update the weights - unlike ESL we normalise the weights
            self.weights = [
                self.weights[i]*np.exp(alpha*(truths[i] != preds[i])) for i in range(len(self.weights))]
            self.weights = [self.weights[i] /
                            sum(self.weights) for i in range(len(self.weights))]

        # 3. The full classifier is now stored in self.alphas and self.classifiers

    def predict(self, dataset):
        preds = []
        for item in dataset:
            item = item[1][:-1]
            pred = sum(self.classifiers[i].treePredict(
                item)*self.alphas[i] for i in range(len(self.alphas)))
            preds.append(np.sign(pred))
        return preds

    def predictSamme(self, dataset):
        preds = []
        for item in dataset:
            item = item[1][:-1]
            predList = [sum((self.classifiers[i].treePredict(
                item) == feature)*self.alphas[i] for i in range(len(self.alphas))) for feature in self.features]
            pred = max(enumerate(predList), key=lambda x: x[1])[0]
            preds.append(pred)
        return preds
