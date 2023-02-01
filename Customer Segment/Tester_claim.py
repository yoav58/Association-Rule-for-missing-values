from AprioriForNullMachineLearningPrediction_Customer import association_rule_method
from SimpleMachineLearningPrediction import simple_method
from Helper import HelperMethods

t = [0.5, 0.6, 0.7, 0.8, 0.9]
arm = lambda t: association_rule_method(t)
sm = lambda: simple_method()
HelperMethods.test(arm,sm,t,'Accuracy',10)