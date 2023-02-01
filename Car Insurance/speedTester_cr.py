from AprioriForNullMachineLearningPrediction import association_rule_method
from SimpleMachineLearningPrediction import simpleMethod
from Helper import HelperMethods



t = [0.6]
arm = lambda t: association_rule_method(t)
sm = lambda: simpleMethod()
HelperMethods.compareSpeed(arm, sm, t, 'Speed', 1)