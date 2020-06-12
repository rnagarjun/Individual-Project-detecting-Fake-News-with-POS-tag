import pandas as pd
import numpy as np
import statsmodels.stats.outliers_influence as oi
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

def num_array():
    df = pd.read_csv('preprocessed_data.csv')


    df['label'] = pd.to_numeric(df['label'])
    X_train = df.as_matrix()
    #df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)

    #final = np.corrcoef(X_train, rowvar=False)
    #np.savetxt("foo.csv", final, delimiter=",")
    df.rename(columns = {'PRP$':'PRP2', 'WP$':'WP2'}, inplace = True)
    features = '+'.join(df.columns)
    #print(features)
    y, X = dmatrices('label ~' + features, df, return_type='dataframe')

    vif = pd.DataFrame()
    vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
    vif.to_csv('vif.csv')
    np.savetxt("foo.csv", final, delimiter=",")
    print(vif)


if __name__ == '__main__':
    num_array()
