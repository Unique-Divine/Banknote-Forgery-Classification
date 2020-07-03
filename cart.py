import numpy as np
from uncertainty import entropy, gini
# Type imports
from numpy import ndarray

class Node:
    """A node in the decision tree. Each node of a tree has attributes such as
    left and right children, the data and class labels to split on, a 
    threshold value, threshold feature index, and an uncertainty measure.
    
    Args:
        data (Array): Input data, X.
        labels (Array): Target vectors, Y.
        depth (int): Tree depth. The root node starts at depth 0.
    
    Attributes:
        left (Node): Left child node. Defaults to None.
        right (Node): Right child node. Defaults to None.
        threshold (float):
        threshold_idx (int): Index in the dataframe for a particular feature.
        feature_col_idx (int): Feature column number 
        label: 
        uncertainty: Uncertainty measure for the node. 
    """
    def __init__(self, data: ndarray, labels: ndarray, depth: int):
        self.left = None
        self.right = None

        self.data = data
        self.labels = labels
        self.depth = depth

        self.threshold: float = None # threshold value
        self.threshold_idx: int = None # threshold index
        self.feature_col_idx = None # feature as a NUMBER (column number)
        self.label = None # y label
        self.uncertainty = None # uncertainty value

class DecisionTree:
    """
    Args:
        K (int): number of features to split on 
    """
    def __init__(self, K=5, verbose=False):

        self.root = None
        self.K = K
        self.verbose = verbose

    def buildTree(self, data: ndarray, labels: ndarray):
        """Builds tree for training on data. Recursively called _buildTree"""
        self.root = Node(data = data, labels = labels, depth = 0)
        if self.verbose:
            print("Root node shape: ", data.shape, labels.shape)
        self._buildTree(self.root)

    def _buildTree(self, node: Node):
        # Get uncertainty measure and feature threshold
        node.uncertainty = self.get_uncertainty(node.labels)
        self.set_feature_threshold(node)
        
        # Sort feature for return
        idx = node.data[:, node.feature_col_idx].argsort()
        node.data = node.data[idx]
        node.labels = node.labels[idx]      
        
        label_distribution = np.bincount(node.labels)       
        if self.verbose:
            print(f"Node uncertainty: {node.uncertainty}")
        
        # Split left and right if threshold is not a minima of the feature 
        if (node.threshold_idx == 0 or 
            node.threshold_idx == node.data.shape[0] or
            len(label_distribution) == 1):
            node.label = (node.labels[0] if len(label_distribution) == 1 
                          else np.argmax(label_distribution) )
        else:
            node.left = Node(node.data[:node.threshold_idx],
                             node.labels[:node.threshold_idx],
                             node.depth + 1)
            node.right = Node(node.data[node.threshold_idx:],
                              node.labels[node.threshold_idx:],
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

    def _predict(self, data_pt, node: Node):
        feature = node.feature_col_idx
        threshold = node.threshold
        if node.label is not None:
            return node.label
        elif data_pt[node.feature_col_idx] < node.threshold:
            return self._predict(data_pt, node.left)
        elif data_pt[node.feature_col_idx] >= node.threshold:
            return self._predict(data_pt, node.right)

    def set_feature_threshold(self, node: Node) -> None:
        """Finds the feature corresponding to the largest information
        gain, then updates node.threshold, node.threshold_idx, and 
        node.feature_col_idx, a number representing the feature.
        
        Returns: 
            None
        """
        node.threshold = 0
        node.threshold_idx = 0
        node.feature_col_idx = 0

        gain, idx, feature = 0, 0, 0 
        for k in range(len(node.data[0])):
            d = np.argsort(node.data[:, k])
            node.labels = node.labels[d]
            node.data = node.data[d]
            for j in range(len(node.data)):
                if self.getInfoGain(node, j) > gain:
                    gain, idx, feature = self.getInfoGain(node, j), j, k
        feat = np.argsort(node.data[:, feature])
        node.data = node.data[feat]
        node.labels = node.labels[feat]
        
        node.threshold = node.data[idx, feature]
        node.threshold_idx = idx
        node.feature_col_idx = feature 

    def getInfoGain(self, node: Node, split_idx: int) -> float:
        """Gets the information gain at a given node (decision) in the tree.

        Args:
            node (Node): The node at which information gain will be evaluated.
            split_idx (int): Index in the feature column that we 
                split the classes on

        Returns:
            infogain (float): information gain
        """
        left_uncertainty = self.get_uncertainty(node.labels[:split_idx]) 
        right_uncertainty = self.get_uncertainty(node.labels[split_idx:])
        
        n = len(node.labels)
        w1 = left_uncertainty * (split_idx + 1) / n
        w2 = right_uncertainty * (n - (split_idx + 1)) / n
        
        start_entropy = self.get_uncertainty(node.labels)
        conditional_entropy = w1 + w2 
        infogain: float = start_entropy - conditional_entropy
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

    def print_tree(self):
        """Prints the tree including threshold value and feature name"""
        self._print_tree(node = self.root)

    def _print_tree(self, node: Node):
        if node is not None:
            if node.label is None:
                print("\t" * node.depth, "(%d, %d)" 
                      % (node.threshold, node.feature_col_idx))
            else:
                print("\t" * node.depth, node.label)
            self._print_tree(node.left)
            self._print_tree(node.right)

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