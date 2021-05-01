import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

class Node:
    def __init__(self):
        # links to the left and right child nodes
        self.right = None
        self.left = None

        # derived from splitting criteria
        self.column = None
        self.threshold = None

        # probability for object inside the Node to belong for each of the given classes
        self.probas = None
        # depth of the given node
        self.depth = None

        # if it is the root Node or not
        self.is_terminal = False


class DecisionTreeClassifier:
    def __init__(self, max_depth=3, min_samples_leaf=1, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

        # Decision tree itself
        self.Tree = None

    def nodeProbas(self, y):
        '''
        Calculates probability of class in a given node
        '''

        probas = []

        # for each unique label calculate the probability for it
        for one_class in self.classes:
            proba = y[y == one_class].shape[0] / y.shape[0]
            probas.append(proba)
        return np.asarray(probas)

    def gini(self, probas):
        '''
        Calculates gini criterion
        '''

        return 1 - np.sum(probas ** 2)

    def calcImpurity(self, y):
        '''
        Wrapper for the impurity calculation. Calculates probas first and then passses them
        to the Gini criterion
        '''

        return self.gini(self.nodeProbas(y))

    def calcBestSplit(self, X, y):
        '''
        Calculates the best possible split for the concrete node of the tree
        '''

        bestSplitCol = None
        bestThresh = None
        bestInfoGain = -999

        impurityBefore = self.calcImpurity(y)

        # for each column in X
        for col in range(X.shape[1]):
            x_col = X[:, col]

            # for each value in the column
            for x_i in x_col:
                threshold = x_i
                y_right = y[x_col > threshold]
                y_left = y[x_col <= threshold]

                if y_right.shape[0] == 0 or y_left.shape[0] == 0:
                    continue

                # calculate impurity for the right and left nodes
                impurityRight = self.calcImpurity(y_right)
                impurityLeft = self.calcImpurity(y_left)

                # calculate information gain
                infoGain = impurityBefore
                infoGain -= (impurityLeft * y_left.shape[0] / y.shape[0]) + (
                            impurityRight * y_right.shape[0] / y.shape[0])

                # is this infoGain better then all other?
                if infoGain > bestInfoGain:
                    bestSplitCol = col
                    bestThresh = threshold
                    bestInfoGain = infoGain

        # if we still didn't find the split
        if bestInfoGain == -999:
            return None, None, None, None, None, None

        # making the best split

        x_col = X[:, bestSplitCol]
        x_left, x_right = X[x_col <= bestThresh, :], X[x_col > bestThresh, :]
        y_left, y_right = y[x_col <= bestThresh], y[x_col > bestThresh]

        return bestSplitCol, bestThresh, x_left, y_left, x_right, y_right

    def buildDT(self, X, y, node):
        '''
         Recursively builds decision tree from the top to bottom
         '''

        # checking for the terminal conditions

        if node.depth >= self.max_depth:
            node.is_terminal = True
            return

        if X.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return

        if np.unique(y).shape[0] == 1:
            node.is_terminal = True
            return

        # calculating current split
        splitCol, thresh, x_left, y_left, x_right, y_right = self.calcBestSplit(X, y)

        if splitCol is None:
            node.is_terminal = True

    def fit(self, X, y):
            '''
            Standard fit function to run all the model training
            '''

            if type(X) == pd.DataFrame:
                X = np.asarray(X)

            self.classes = np.unique(y)
            # root node creation
            self.Tree = Node()
            self.Tree.depth = 1
            self.Tree.probas = self.nodeProbas(y)

            self.buildDT(X, y, self.Tree)

    def predictSample(self, x, node):
        '''
        Passes one object through decision tree and return the probability of it to belong to each class
        '''

        # if we have reached the terminal node of the tree
        if node.is_terminal:
            return node.probas

        if x[node.column] > node.threshold:
            probas = self.predictSample(x, node.right)
        else:
            probas = self.predictSample(x, node.left)

        return probas

    def predict(self, X):
        '''
        Returns the labels for each X
        '''

        if type(X) == pd.DataFrame:
            X = np.asarray(X)

        predictions = []
        for x in X:
            pred = np.argmax(self.predictSample(x, self.Tree))
            predictions.append(pred)

        return np.asarray(predictions)

df = pd.read_csv('WINE_DATA.csv')
print(df.head())
print(df.shape)

# One-hot encode the data using pandas get_dummies
######################## A handy way To display columns fully #############
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)
###########################################################################
print(df.head())
features = pd.get_dummies(df)
# print(features.corr())
# Display the first 5 rows of the last 12 columns
# print(features.iloc[:,5:].head(5))
print(features.shape)
print(features.head())
labels = np.array(features['Wine_Type'])
features= features.drop('Wine_Type', axis = 1)
feature_list = list(features.columns)
print(feature_list)
features = np.array(features)

## Run model training
dt = DecisionTreeClassifier()
print(dt.fit(features, labels))

X = [3,12.96,3.45,2.35,18.5,106,1.39,.7,.4,.94,5.28,.68,1.75,1.0,675]
print(dt.predict(X))

def solution(WINE_DATA, test_user):
    # Your code goes here
    return WINE_TYPE


TEST_USERS = [
    [3, 12.96, 3.45, 2.35, 18.5, 106, 1.39, .7, .4, .94, 5.28, .68, 1.75, 1.0, 675],
    [3, 13.17, 2.59, 2.37, 20, 120, 1.65, .68, .53, 1.46, 9.3, .6, 1.62, 1.0, 840],
    [1, 13.16, 2.36, 2.67, 18.6, 101, 2.8, 3.24, .3, 2.81, 5.68, 1.03, 3.17, 1.0, 1185],
    [2, 12.37, .94, 1.36, 10.6, 88, 1.98, .57, .28, .42, 1.95, 1.05, 1.82, 1.0, 520]
]

# final_dataset=pd.get_dummies(final_dataset,drop_first=True)

# final_dataset= df.drop(['Year','OD280/OD315_of_diluted_wines'], axis=1, inplace=True)
# print(df.head())

# solution(WINE_DATA, TEST_USER[1:])

# for TEST_USER in TEST_USERS:
#     WINE_TYPE = solution(WINE_DATA, TEST_USER[1:])
    # if WINE_TYPE == TEST_USER[0]:
    #     print(True)
    # else:
    #     print(False)
