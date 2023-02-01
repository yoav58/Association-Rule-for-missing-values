import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import time


class HelperMethods:

    #####################################################
    # Name: drop_attributes
    # Description: this function delete attributes
    ####################################################
    @staticmethod
    def drop_attributes(data,attributes):
        for attribute in attributes:
            data = data.drop(attribute,axis=1)
        return data

    #####################################################
    # Name: rangeMeanDictionary
    # Description: this function create dictionary of
    # label and mean, for example {low:220,Medium:405,High:700}
    ####################################################
    @staticmethod
    def rangeMeanDictionary(df, range, labels, attribute_name):
        dictionary = {}
        min_value = 0
        max_value = 1
        for label in labels:
            df_mean = df.loc[ (df[attribute_name] >= range[min_value]) & (df[attribute_name] <= range[max_value])]
            mean = df_mean[attribute_name].mean()
            dictionary[label] = mean
            min_value +=1
            max_value +=1
        return dictionary

    ###############################################################
    # Name: convertToCategorial
    # Description: convert to categorical by using cut function.
    ###############################################################
    @staticmethod
    def convertToCategorial(data, bins, labels, attributeName):
        return pd.cut(data[attributeName],bins=bins,labels=labels)


    ###############################################################
    # Name: restoreNullValues
    # Description: since some operation must do without null values,
    # this function help to restore them after delete them.
    ###############################################################
    @staticmethod
    def restoreNullValues(df,origin_df,categories):
        for category in categories:
            df.loc[origin_df.vehicleType.isnull(), df.columns.str.startswith(category)] = np.nan

    ###############################################################
    # Name: remove_outliers
    # Description: remove outliers using IQR
    ###############################################################
    @staticmethod
    def remove_outliers(df):
        q1 = df.quantile(.1, numeric_only=True)
        q3 = df.quantile(.9, numeric_only=True)
        iqr = q3 - q1
        return df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]

    ####################################################################################################
    # Name: add_collumns
    # Description: because using getdummy can cause to the training attributes and the test be diffrent
    # this method helps to add columns which appear in one df but not in the other
    ###################################################################################################
    @staticmethod
    def add_collumns(data,columns):
        data_columns = set(data.columns)
        for column in columns:
            if(column not in data_columns):
                data[column] = 0

    ####################################################################################################
    # Name: show_correlations
    # Description: simply show the correlations.
    ###################################################################################################
    @staticmethod
    def show_correlations(data_frame,categorical_columns):
        encoder = LabelEncoder()
        encoded_dtf = data_frame.copy()
        encoded_dtf[categorical_columns] = encoded_dtf[categorical_columns].apply(encoder.fit_transform)
        corr = encoded_dtf.corr()
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=corr.index,
                y=corr.index,
                z=corr,
                text=corr.values,
                texttemplate='%{text:.2f}'
            )
        )
        fig.update_layout(width=1000, height=1000, title="Heatmap correlations")
        fig.show()
        return corr


    ####################################################################################################
    # Name: equal_convertToCategorical
    # Description: convert to categorical where each bin have the same amount of appearances.
    ###################################################################################################
    @staticmethod
    def equal_convertToCategorical(df,labels,amountofbins,attribute):
        new_attribute = attribute + '_cat'
        df[new_attribute] =  pd.cut(df[attribute],amountofbins,labels=labels)
        means = df.groupby(new_attribute)[attribute].mean()
        dic = means.to_dict()
        df = df.drop(new_attribute,axis=1)
        df[attribute] = pd.cut(df[attribute],amountofbins,labels=labels)
        return df,dic


    ####################################################################################################
    # Name: convert_with_astype
    # Description: convert to categorical with astype
    ###################################################################################################
    @ staticmethod
    def convert_with_astype(data,attributes):
        for att in attributes:
            data[att] = data[att].astype('category')

    ####################################################################################################
    # Name: convert_to_num
    # Description: convert to num with astype
    ###################################################################################################
    @staticmethod
    def convert_to_num(data,attributes):
        for att in attributes:
            data[att] = data[att].astype('float64')


    ####################################################################################################
    # Name: test
    # arguments:
    # tr: the threshold
    # test_type = what to test, the accuracy or the r2_score
    # n_itr: number of iterations to check.
    # Description: this function is to test the accuracy of the association rule method agains the simple
    # method
    ###################################################################################################
    @staticmethod
    def test(association_rule_method, simpleMethod, tr, test_type, n_itr):
        thresholds = tr
        association_rule_results = []
        number_of_iterations = n_itr
        for threshold in thresholds:
            value = []
            for i in range(number_of_iterations):
                print("thresholds number:" + str(threshold) + ", iteration number: " + str(i))
                result = association_rule_method(threshold)
                value.append(result)
            arr = np.array(value)
            value = np.mean(arr)
            association_rule_results.append(value)

        conventional_result_temp = []
        for i in range(number_of_iterations):
            result = simpleMethod()
            conventional_result_temp.append(result)
        arr = np.array(conventional_result_temp)
        conventional_result_temp = arr.mean()
        conventional_result = [conventional_result_temp for i in range(5)]

        # set width of bar
        barWidth = 0.25
        fig = plt.subplots(figsize=(12, 8))

        # set height of bar
        IT = association_rule_results
        CSE = conventional_result

        # Set position of bar on X axis
        br1 = np.arange(len(IT))
        br2 = [x + barWidth for x in br1]

        # Make the plot
        plt.bar(br1, IT, color='r', width=barWidth,
                edgecolor='grey', label='Asocciation Rule Method')
        plt.bar(br2, CSE, color='b', width=barWidth,
                edgecolor='grey', label='Conventional Method')

        # Adding Xticks

        s_thresholds = [str(t) for t in thresholds]
        plt.xlabel('Threshold', fontweight='bold', fontsize=15)
        plt.ylabel(test_type, fontweight='bold', fontsize=15)
        plt.xticks([r + barWidth for r in range(len(IT))],
                  s_thresholds)

        plt.legend()
        plt.show()


    @staticmethod
    def compareSpeed(association_rule_method, simpleMethod, tr, test_type, n_itr):
        thresholds = tr
        association_rule_results = []
        number_of_iterations = n_itr
        for threshold in thresholds:
            value = []
            for i in range(number_of_iterations):
                print("thresholds number:" + str(threshold) + ", iteration number: " + str(i))
                t0 = time.time()
                association_rule_method(threshold)
                t1 = time.time()
                result = (t1-t0)
                value.append(result)
            arr = np.array(value)
            value = np.mean(arr)
            association_rule_results.append(value)

        conventional_result_temp = []
        for i in range(number_of_iterations):
            st0 = time.time()
            simpleMethod()
            st1 = time.time()
            result = (st1-st0)  # since the time it take is super low.
            conventional_result_temp.append(result)
        arr = np.array(conventional_result_temp)
        conventional_result_temp = arr.mean()
        conventional_result = [conventional_result_temp for i in range(len(tr))]

        # set width of bar
        barWidth = 0.25
        fig = plt.subplots(figsize=(12, 8))

        # set height of bar
        IT = association_rule_results
        CSE = conventional_result

        # Set position of bar on X axis
        br1 = np.arange(len(IT))
        br2 = [x + barWidth for x in br1]

        # Make the plot
        plt.bar(br1, IT, color='r', width=barWidth,
                edgecolor='grey', label='Asocciation Rule Method')
        plt.bar(br2, CSE, color='b', width=barWidth,
                edgecolor='grey', label='Conventional Method')

        # Adding Xticks

        s_thresholds = [str(t) for t in thresholds]
        plt.xlabel('Threshold', fontweight='bold', fontsize=15)
        plt.ylabel(test_type, fontweight='bold', fontsize=15)
        plt.xticks([r + barWidth for r in range(len(IT))],
                  s_thresholds)
        ymin = 0
        ymax = 10
        # plt.ylim(ymin,ymax)

        plt.legend()
        plt.show()






