import numpy as np

# Entropy and Gini Calculations
def entropy(y_labels):
    _, fraud_count = np.unique(y_labels, return_counts=True)
    p_i = fraud_count / fraud_count.sum() # probability (array) of each class
    entropy = np.sum(p_i * -np.log2(p_i))
    return entropy

def gini(y_labels):
    _, fraud_count = np.unique(y_labels, return_counts=True)
    p_i = fraud_count / fraud_count.sum() # probability (array) of each class
    gini = 1 - np.sum(p_i**2)
    return gini
    
def total_entropy(partition_0, partition_1):
    n = len(partition_0) + len(partition_1)
    prob_part0 = len(partition_0) / n # probability of partition 0
    prob_part1 = len(partition_1) / n # probabiltiy of partition 1 
    tot_entropy = (prob_part0 * entropy(partition_0)
        + prob_part1 * entropy(partition_1) )
    return tot_entropy