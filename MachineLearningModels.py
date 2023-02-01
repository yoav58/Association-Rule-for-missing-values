from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MachineLearningModels:
    @staticmethod
    def LogisticRegressionModel(train,test,target):
        # start machine learning prediction
        X_train = train.copy().drop(target, axis=1)
        X_test = test.copy().drop(target, axis=1)
        Y_train = train[target]
        Y_test = test[target]
        # using logistic regression model since the attribute OUTCOME is binary
        clf = LogisticRegression(max_iter=10000)
        clf.fit(X_train, Y_train)
        prediction = clf.predict(X_test)
        return prediction, Y_test

