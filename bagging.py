# Bagging is used to reduce the high variance of decision trees, as different training sets (sampled randomly from within one large training set) can lead to dramatically different trees.
# Basic theoretical idea: given n independent X_i with constant variance s**2, sample once from each and take mean to get sample mean X. This has variance s**2/n

# We use k splitting of the data for cross-validation to establish error terms.
# For computational efficiency, for observation i you could average across all trees where observation i is out of bag to generate a predicted response.
# This requires some slight refactoring of the code to achieve

# This is a close-to pure python implementation using the numpy random sub module for random sampling (you could use the random module).
# We'll also use pandas to read in a csv file, so we don't have to declare our data in-line.
# There is no ability to read csvs in pure python, but you could use the csv module in the Python Standard Library as a closest-to-pure alternative.
# The assumption of this implementation is that the data structure will be a list of lists combining the observations at p features with a classifier.
# So each sublist is p+1 length - p features followed by the classifier for each observation.
# We use the gini index to split based on optimal classifier at each node in the tree.

# An optional rf parameter can be supplied into the bagging/scoreBagging functions to implement a random forest, which restricts the feature choice at a split point to one of sqrt(p) features, randomly chosen.
# An optional ratio parameter can be supplied to reduce the size of the sample used in the bagging algorithm (by default it is equal to the number of the observations, so asymptotically containing 1-e**-1 proportion of the observations)


import numpy as np
import pandas as pd


# Helper function for Gini Index based on split of the data
def gini(groups, classes):
    n = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        if not group:
            continue
        score = 0.0
        for cla in classes:
            p = [row[-1] for row in group].count(cla)/len(group)
            score += p ** 2
        gini += (1.0-score)*(len(group)/n)
    return gini


# Split dataset based on value of an attribute
def splitAttr(dataset, attr, val):
    x, y = [], []
    for item in dataset:
        if item[attr] < val:
            x.append(item)
        else:
            y.append(item)
    return x, y


# Find the best attribute to split a dataset on and return an object that acts as a node in a decision tree.
# Takes an optional rf argument that implements the random forest
def split(dataset, rf):
    # If random forest,randomly pick sqrt(len(p)) features from feature list p for which to consider list
    allowList = np.random.randint(0, len(dataset[0]), round(
        np.sqrt(len(dataset[0])))) if rf else list(range(len(dataset[0])-1))

    # Potential outcomes
    outs = list({item[-1] for item in dataset})
    giniMin = 10e6
    # Loop over data items and indices to find best gini index
    for item in dataset:
        for i in range(len(dataset[0])-1):
            if i in allowList:
                groups = splitAttr(dataset, i, item[i])
                gin = gini(groups, outs)
                if gin < giniMin:
                    giniMin, attr, val, nodes = gin, i, item[i], groups
    # Return a dict object: here we keep children so we can move down the tree, the attribute on which this node splits, and the value that determines which child to go to
    return {"attr": attr, "val": val, "children": nodes}


# Create end node with classifier based on majority vote. Used when depth/node size constraints are hit.
def end(list):
    outs = [item[-1] for item in list]
    return max(set(outs), key=outs.count)


# Split a tree at a node
def treeSplit(node, min, max, depth, rf):
    # Takes min (size), max (depth), and current depth inputs.
    # Get the groups of data on each branch of the tree at current node
    l, r = node["children"]
    if not l or not r:
        # If either empty, terminate at this node based on what remains in other
        node["l"], node["r"] = end(l+r), end(l+r)
        return
    elif depth >= max:
        # If too deep, terminate each node in place
        node["l"], node["r"] = end(l), end(r)
        return
    else:
        # If node too small, end there, otherwise split further
        if len(l) <= min:
            node["l"] = end(l)
        else:
            node["l"] = split(l, rf)
            treeSplit(node["l"], min, max, depth+1, rf)
        if len(r) <= min:
            node["r"] = end(r)
        else:
            node["r"] = split(r, rf)
            treeSplit(node["r"], min, max, depth+1, rf)


# Create a decision tree based on a  dataset with min branch size and max depth
def createTree(dataset, min, max, rf):
    start = split(dataset, rf)
    treeSplit(start, min, max, 1, rf)
    return start


# Use the tree to make a prediction, picks route based on value, then checks if node is an end or not, either retrieving the value or going deeper.
def treePredict(node, item):
    if item[node["attr"]] < node["val"]:
        return treePredict(node["l"], item) if isinstance(node["l"], dict) else node["l"]
    else:
        return treePredict(node["r"], item) if isinstance(node["r"], dict)else node["r"]


# Make an individual bagging prediction
def baggingPred(trees, item):
    outs = [treePredict(tree, item) for tree in trees]
    return max(set(outs), key=outs.count)


# The bagging algorithm
def bagging(train, test, min, max, n, ratio=1.0, rf=False):
    # Create n trees, predict outcome for each item in the test data using these trees
    trees = [createTree(subsample(train, ratio)[0], min, max, rf)
             for _ in range(n)]
    return [baggingPred(trees, item) for item in test]


# Split the dataset for k-fold cross
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
        test_set = [item[:-1] for item in fold]
        preds = bagging(train_set, test_set, min, max, n_trees, ratio, rf)
        truth = [item[-1]for item in fold]
        total += accuracy(truth, preds)
    return total/k_folds
