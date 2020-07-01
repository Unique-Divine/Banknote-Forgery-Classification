import numpy as np
from uncertainty import entropy, gini

class Node:
    """Each node of our decision tree will hold values such as left
    and right children, the data and labels being split on, the 
    threshold value & index in the dataframe for a particular 
    feature, and the uncertainty measure for this node"""
    def __init__(self, data, labels, depth):
        """
        Args: 
            data: Input data, X.
            labels: Target vectors, Y. 
            depth: Tree depth. 
        """
        self.left = None
        self.right = None

        self.data = data
        self.labels = labels
        self.depth = depth

        self.threshold = None # threshold value
        self.threshold_index = None # threshold index
        self.feature = None # feature as a NUMBER (column number)
        self.label = None # y label
        self.uncertainty = None # uncertainty value

class DecisionTree:
    def __init__(self, K=5, verbose=False):
        """
        K: number of features to split on 
        """
        self.root = None
        self.K = K
        self.verbose = verbose

    def buildTree(self, data, labels):
        """Builds tree for training on data. Recursively called _buildTree"""
        self.root = Node(data, labels, 0)
        if self.verbose:
            print("Root node shape: ", data.shape, labels.shape)
        self._buildTree(self.root)

    def _buildTree(self, node):
        # get uncertainty measure and feature threshold
        node.uncertainty = self.get_uncertainty(node.labels)
        self.get_feature_threshold(node)
        
        # sort feature for return
        index = node.data[:, node.feature].argsort()
        node.data = node.data[index]
        node.labels = node.labels[index]      
        
        label_distribution = np.bincount(node.labels)       
        if self.verbose:
            print("Node uncertainty: %f" % node.uncertainty)
        
        # Split left and right if threshold is not a minima of the feature 
        if (node.threshold_index == 0 or 
            node.threshold_index == node.data.shape[0] or
            len(label_distribution) == 1):
            node.label = (node.labels[0] if len(label_distribution) == 1 
                          else np.argmax(label_distribution) )
        else:
            node.left = Node(node.data[:node.threshold_index],
                             node.labels[:node.threshold_index],
                             node.depth + 1)
            node.right = Node(node.data[node.threshold_index:],
                              node.labels[node.threshold_index:],
                              node.depth + 1)
            node.data = None
            node.labels = None
            
            # If the node is in the last layer of tree, assign predictions
            if node.depth == self.K:
                if len(node.left.labels) == 0:
                    node.right.label = np.argmax(np.bincount(node.right.labels))
                    node.left.label = 1 - node.right.label
                elif len(node.right.labels) == 0:
                    node.left.label = np.argmax(np.bincount(node.left.labels))
                    node.right.label = 1 - node.left.label
                else:
                    node.left.label = np.argmax(np.bincount(node.left.labels))
                    node.right.label = np.argmax(np.bincount(node.right.labels))
                return

            else: # Otherwise continue training the tree
                self._buildTree(node.left)
                self._buildTree(node.right)

    def predict(self, data_pt):
        return self._predict(data_pt, self.root)

    def _predict(self, data_pt, node):
        feature = node.feature
        threshold = node.threshold
        if node.label is not None:
            return node.label
        elif data_pt[node.feature] < node.threshold:
            return self._predict(data_pt, node.left)
        elif data_pt[node.feature] >= node.threshold:
            return self._predict(data_pt, node.right)

    def get_feature_threshold(self, node):
        """ This function finds the feature that gives the largest information
        gain, then updates node.threshold, node.threshold_index, and 
        node.feature, a number representing the feature.
        
        return: None
        """
        node.threshold = 0
        node.threshold_index = 0
        node.feature = 0

        gain, index, feature = 0, 0, 0 
        for k in range(len(node.data[0])):
            d = np.argsort(node.data[:, k])
            node.labels = node.labels[d]
            node.data = node.data[d]
            for j in range(len(node.data)):
                if self.getInfoGain(node, j) > gain:
                    gain, index, feature = self.getInfoGain(node, j), j, k
        feat = np.argsort(node.data[:, feature])
        node.data = node.data[feat]
        node.labels = node.labels[feat]
        
        node.threshold = node.data[index, feature]
        node.threshold_index = index
        node.feature = feature 

    def getInfoGain(self, node, split_index):
        """Gets information gain at a given node (decision) in the tree.

        Args:
            node (Node): The node at which the information gain will be evaluated.
            split_index (int): Index in the feature column that we 
                split the classes on

        Returns:
            infogain (float): information gain
        """
        left_uncertainty = self.get_uncertainty(node.labels[:split_index]) 
        right_uncertainty = self.get_uncertainty(node.labels[split_index:])
        
        n = len(node.labels)
        w1 = left_uncertainty * (split_index + 1) / n
        w2 = right_uncertainty * (n - (split_index + 1)) / n
        
        start_entropy = self.get_uncertainty(node.labels)
        conditional_entropy = w1 + w2 
        infogain = start_entropy - conditional_entropy
        return infogain

    def get_uncertainty(self, labels, metric="gini"):
        """Get uncertainty value using either entropy OR gini index. 
        
        Args:
            labels (ndarray): Training labels, or targets .
            metric (str, optional): [description]. Defaults to "gini".

        Returns: uncertainty (float)
        """
        
        if labels.shape[0] == 0:
            return 1
        if metric =="gini":    
            uncertainty = gini(labels)
        if metric == "entropy":
            uncertainty = entropy(labels)
        
        return uncertainty

    def printTree(self):
        """Prints the tree including threshold value and feature name"""
        self._printTree(self.root)

    def _printTree(self, node):
        if node is not None:
            if node.label is None:
                print("\t" * node.depth, "(%d, %d)" 
                      % (node.threshold, node.feature))
            else:
                print("\t" * node.depth, node.label)
            self._printTree(node.left)
            self._printTree(node.right)

    def tree_evaluate(self, X_train, labels, X_test, y_test, verbose=False): 
        #---------
        # Training
        #---------
        n = X_train.shape[0] # number of training samples
        count = 0 # number of accurate predictions
        
        # for all training samples
        for i in range(n):
            # if the prediction matches the target,
            if self.predict(X_train[i]) == labels[i]:
                count += 1
        
        accuracy_train = (count / n) * 100
        if verbose == True:
            print(f"Training accuracy: {(count / n) * 100:.1f}% "
              + f"on {n} samples." )
        #---------
        # Testing
        #---------
        n = X_test.shape[0] # number of test samples
        count = 0 # number of accurate predictions
        
        # for all test samples
        for i in range(n):
            # if the prediction matches the target,
            if self.predict(X_test[i]) == y_test[i]:
                # 
                count += 1
        accuracy_test = (count / n) * 100
        if verbose == True:
            print(f"Test accuracy: {(count / n) * 100:.1f}% "
                  + f"on {n} samples." )

        return accuracy_train, accuracy_test