#https://github.com/Syed-Al-Khwarizmi/decision-tree-scratch-implementation


import sys
import pandas as pd
import numpy as np

"""
The code is theoretically inspired from the following references:
Artificial Intelligence: A Modern Approach, 3rd Edition. (From pages 697-707)
https://en.wikipedia.org/wiki/Decision_tree

Throughout the comments, the said book shall be referred to as AI:AMA
"""


"""
Calculating the entropy, and information gain of an attribute in the decision trees
is a primary operation, which tell us that how important a specific attribute is.
AI:AMA Page no. 704 explains that how these two functions help in choosing what
attribute is placed at the top of the tree, and what goes as the leaf.
"""
def calc_entropy(df, target):
    """
    :param df: The Pandas DataFrame containing the table data
    :param target: The attribute for which we're calculating the entropy
    :return: The entropy of the target attribute
    """
    freq = df[target].value_counts()
    entropy = sum(-freq * 1.0 / len(df) * np.log2(1.0 * freq / len(df)))
    return entropy


def IG(df, attr, target): #IG: Information Gain
    """
    :param df: Pandas DataFrame data in table form
    :param attr: Attribute on which the IG has to be calculated
    :param target: Target attribute
    :return: Information gain which is calculated by splitting the data based on the attribute attr
    """
    #Counting the number of appearance of the values associated by that attribute attr
    freq = df[attr].value_counts()
    sum_counts = sum(freq)
    subset_entropy = 0.0

    # Calculation of sum of entropy for each subset of records weighted by
    # their probability of occurring in the training set.
    for value in freq.index:
        probability = freq.get(value) / sum_counts
        subset = df[df[attr] == value] #Splitting the data to get the subset
        subset_entropy += probability * calc_entropy(subset, target)


        # Calculation of information gain
    return (calc_entropy(df, target) - subset_entropy)





def most_frequent_item(df, target):
    """
    :param df: Input DataFrame
    :param target: Target classification value (Kind of Owl)
    :return: Returns the most frequent item in the data frame
    """
    most_frequent_value = df[target].value_counts().idxmax()
    return most_frequent_value


def get_attribute_values(df, attribute):
    """
    :param df: DataFrame
    :param attribute: Attribute for which the unique items have to be returned
    :return: All unique values for a specific attribute
    """
    return df[attribute].unique()


def best_attribute(df, attributes, target, splitting_heuristic):
    """
    Iterates through all the attributes and returns the attribute with the best splitting_heuristic
    """
    best_value = 0.0
    best_gain = 0.0
    best_attr = None

    for attr in attributes:
        gain = splitting_heuristic(df, attr, target)
        if (gain >= best_gain and attr != target):
            best_gain = gain
            best_attr = attr

    return best_attr


def create_decision_tree(df, attributes, target, splitting_heuristic_func):
    """
    Input : df-Pandas Data Frame, attributes-list of features, target-Target Feature, splitting_heuristic_func-Function to find best feature for splitting
    Returns a new decision tree based on the examples given.
    """
    vals = df[target]
    #print ("vals", vals)
    default = most_frequent_item(df, target)
    #print ("default", default)

    # If the dataset is empty or there are no independent features
    if target in attributes:
        attributes.remove(target)
    if df.empty or len(attributes) == 0:
        return default
    # If all the target values are same, return that classification value.
    elif len(vals.unique()) == 1:
        return default
    else:
        # Choose the next best attribute to best classify our data based on heuristic function
        best_attr = best_attribute(df, attributes, target, splitting_heuristic_func)

        #print ("best_attr", best_attr)

        # Create an empty new decision tree/node with the best attribute and an empty
        # dictionary object
        tree = {best_attr: {}}

        # Create a new decision tree/sub-node for each of the values in the best attribute field
        for val in get_attribute_values(df, best_attr):
            #print ("tree", tree)
            #print ("val", val)
            # Create a subtree for the current value under the "best" field
            data_subset = df[df[best_attr] == val]
            subtree = create_decision_tree(data_subset,
                                           [attr for attr in attributes if attr != best_attr],
                                           target,
                                           splitting_heuristic_func)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            #print ("subtree {} being added to tree {}".format(subtree, tree))
            tree[best_attr][val] = subtree
            print ("now the tree looks like: {}".format(tree))

    return tree


def get_prediction(data_row, decision_tree):
    """
    This function recursively traverses the decision tree and returns a  classification for the given record.
    """
    # If the current node is a string or integer, then we've reached a leaf node and we can return the classification
    if type(decision_tree) == type("string") or type(decision_tree) == type(1):
        return decision_tree

    # Traverse the tree further until a leaf node is found.
    else:
        #print("dtreekeys", decision_tree.keys())
        attr = list(decision_tree.keys())[0]
        dattr = attr
        sarg = data_row[dattr]
        darg = decision_tree[attr]
        # print("attr, dattr", dattr)
        # print("sarg", sarg)
        # print("darg", darg)
        #print ("attr ", attr)
        #print ("Got ", decision_tree[attr])
        #print ("Got ", data_row[attr])
        if data_row[attr] in decision_tree[attr]:
            #print ("trying to get ", decision_tree[attr][data_row[attr]])
            t = decision_tree[attr][data_row[attr]]
        else:
            t = 'NotIdentified'
        #print("err", t)

        #print(attr, decision_tree[attr], data_row[attr], t)

        return get_prediction(data_row, t)


def predict(tree, predData):
    """
    Input : tree-Tree Dictionary created by create_decision_tree function, predData-Pandas DataFrame on which predictions are made
    Returns a list of predicted values of predData
    """
    predictions = []
    for index, row in predData.iterrows():
        predictions.append(get_prediction(row, tree))


    return predictions



def read_file(filename):
    """
    Tries to read csv file using Pandas
    """
    try:
        data = pd.read_csv(filename, header=0, dtype=str)
    except IOError:
        print("Error: The file {} was not found on this system.".format(filename))
        sys.exit(0)

    return data

def print_tree(tree, str):
    """
    This function recursively crawls through the d-tree and prints it out in a
    more readable format than a straight print of the Python dict object.
    """
    if type(tree) == dict:
        print ("%s%s" % (str, list(tree.keys())[0]))
        for item in tree.values()[0].keys():
            print ("%s\t%s" % (str, item))
            print_tree(tree.values()[0][item], str + "\t")
    else:
        print ("%s\t->\t%s" % (str, tree))

def build_tree(data):
    """
    This function builds tree from Pandas Data Frame with last column as the dependent feature
    """
    attributes = list(data.columns.values)
    print(attributes[0])
    target = attributes[0]
    return create_decision_tree(data,attributes,target,IG)


def shuffle_data(data):
        return data.sample(frac=1).reset_index(drop=True)

def split_data(data):
    train_amount = int(len(data)*0.67)
    train_data = data[:train_amount]
    test_data = data[train_amount:]

    return train_data, test_data

def validate(test, pred):
    total = len(test)
    match = 0
    test = test.values
    for i in range(total):
        print("Expected: {}\tCalculated: {}".format(test[i][4], pred[i]))
        if test[i][4] == pred[i]:
            match += 1
    return (float(match)*100)/total


if __name__ == "__main__":

    avg_acc = 0
    for i in range(10):
        acc = 0
        print ("Cycle no {}".format(i))
        data = read_file("WINE_DATA.csv")
        data = shuffle_data(data)
        train, test = split_data(data)
        #print("data", data)
        #print ("train", train)
        #print ("test", test)
        print("Data Read and Loaded\n")
        print("Building Decision Tree\n")
        tree = build_tree(train)
        #print(print_tree(tree,""))
        predictions = predict(tree, test)
        acc = validate(test, predictions)
        avg_acc += acc
        print("Accuracy: {}".format(acc))

    print ("Total average accuracy: {}".format(avg_acc/10.0))