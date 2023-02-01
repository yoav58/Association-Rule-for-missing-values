from sklearn.impute import KNNImputer



class NullValuesFiller:


    def mustFrequent(df,attribute_names):
        dtf_c = df.copy().dropna()
        for attribute_name in attribute_names:
            most_frequent = dtf_c[attribute_name].value_counts().idxmax()
            df[attribute_name] = df[attribute_name].fillna(most_frequent,inplace=False)


    def mustFrequentTest(df,train,attribute_names):
        train_c = train.copy().dropna()
        for attribute_name in attribute_names:
            most_frequent = train_c[attribute_name].value_counts().idxmax()
            df[attribute_name] = df[attribute_name].fillna(most_frequent,inplace=False)





    def knn(df):
        imputer = KNNImputer(n_neighbors=3)
        imputer.fit(df)
        df_imputed = imputer.transform(df)
        return df_imputed

    def mean(df,df_mean,attributes_names):
        for attribute in attributes_names:
            df[attribute] = df[attribute].fillna(df_mean[attribute].mean())
        return df




