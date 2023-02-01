from AprioriForNullMachineLearning_Titanic import association_rule_method
from SimpleMachineLearningPrediction_titanic import simple_method
from Helper import HelperMethods


t = [0.6]
arm = lambda t: association_rule_method(t)
sm = lambda: simple_method()
HelperMethods.compareSpeed(arm,sm,t,'Speed',1)