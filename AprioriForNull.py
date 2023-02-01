import pandas as pd
from efficient_apriori import apriori


class AprioriForNull:

    ####################################################################################################
    # Name: __fillNullValues
    # Description: this method check if the target attribute value is null, if so fill the value with
    # the rule consequent.
    ###################################################################################################
    def __fillNullValues(self,data, row_index, rules_consequent, rule_index, convert_dictionary, type_dictionary):
        for rule in rules_consequent[rule_index].rhs:
            rule_name = rule[0]
            rule_value = rule[1]
            if (pd.isnull(data.iloc[row_index][rule_name]) == True):
                if (type_dictionary[rule_name] == 'float64' or type_dictionary[rule_name] == 'int64'): # if its type float or int, set the value to the mean.
                    rule_value = convert_dictionary[rule_name][rule_value]
                data.iloc[row_index, data.columns.get_loc(rule_name)] = rule_value


    ####################################################################################################
    # Name: fill_nulls_with_apriori
    # Description: this method find if the rules apply in the row, if so fill the attribute with
    # the "__fillNullValues method"
    ###################################################################################################
    def fill_nulls_with_apriori(self,train_data,categorical_data,rules,convertedDictionary,typeDictionary):
        for rowIndex in range(len(categorical_data)):
            for i in range(len(rules)):
                antecedent_exists = True
                for rule in rules[i].lhs:
                    attribute_name = rule[0]
                    attribute_value = rule[1]
                    if (attribute_value != categorical_data.iloc[rowIndex][attribute_name]):
                        antecedent_exists = False
                        break;
                if (antecedent_exists):
                    self.__fillNullValues(self,data=train_data, row_index=rowIndex, rules_consequent=rules, rule_index=i, convert_dictionary=convertedDictionary, type_dictionary=typeDictionary)

    ####################################################################################################
    # Name: findRules
    # Description: this method is to find the rules.
    ###################################################################################################
    def findRules(data,support_treshold,condifence_treshold):
        categorical_columns = [c for c in data.columns]
        records = data[categorical_columns].to_dict(orient='records')
        transactions = []
        for r in records:
            transactions.append(list(r.items()))
        itemsets, rules = apriori(transactions, min_support=support_treshold, min_confidence=condifence_treshold, output_transaction_ids=False)
        return itemsets, rules
