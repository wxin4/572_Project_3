import os
import joblib
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from scipy import stats
from scipy.stats import entropy
import numpy as np

# traverse to the directory

def directory(dir):
    fileList = []
    for directory, dirs, name in os.walk(dir):
        [fileList.append('{0}/{1}'.format(dir, n)) for n in name]
    return fileList


# get the data for meal and nomeal from the folders
def getData(files):
    df = pd.read_csv(files, names=list(range(31)))
    return fillNans(df[:50])


def getData2(files):
    df = pd.read_csv(files)[:50]
    return fillNans(df)


# remove all the NaN values from the pandas series
def removeNans(pdseries):
    filled_data = []

    # using interpolate to remove the NaN values forwardly
    for series in pdseries:
        filled_data.append(pd.Series(series).interpolate(method='linear', limit_direction='forward').to_list())
    return pd.DataFrame(filled_data)


# fill the NaN values of the dataframe
def fillNans(dataframe):
    # 1. remove NaNs
    cleaned_data = removeNans(dataframe.values.tolist())

    # 2. get the first column and check the NaN value
    first_column = cleaned_data.columns[0]

    # 3. check if it contains null values
    check_isnull = cleaned_data[first_column].isnull()

    # 4. if yes, get its index and create a list of those
    nan_idx_list = cleaned_data[check_isnull].index.tolist()

    # 5. using the median (more accurate) to fill those null values
    values = cleaned_data.median(axis=0).tolist()
    for i in nan_idx_list:
        cleaned_data.loc[i] = values

    return np.array(cleaned_data)


# create a class that can pass the cleaned data into all the features
class features:
    def __init__(self, data):
        self.data = data

    # feature 1. covariance (10 of those to make the accuracy higher)
    def covariance(self):
        cov1 = np.cov(self.data[:, 1:3])
        cov2 = np.cov(self.data[:, 4:6])
        cov3 = np.cov(self.data[:, 7:9])
        cov4 = np.cov(self.data[:, 10:12])
        cov5 = np.cov(self.data[:, 13:15])
        cov6 = np.cov(self.data[:, 16:18])
        cov7 = np.cov(self.data[:, 19:21])
        cov8 = np.cov(self.data[:, 22:24])
        cov9 = np.cov(self.data[:, 25:27])
        cov10 = np.cov(self.data[:, 28:30])

        return (cov1 + cov2 + cov3 + cov4 + cov5 + cov6 + cov7 + cov8 + cov9 + cov10) / 10

    # feature 2. entropy values (calculate the error rate)
    def entropy(self, base=None):
        array = []
        for i in range(len(self.data)):
            value, counts = np.unique(self.data, return_counts=True)
            array.append(entropy(self.data[i, :]))
        np.array(array).reshape((len(self.data), 1))  # reshape to the matrix that can be concatenated
        return np.array(array)

    # feature 3. skewness values (calculate the bias rate) using the built-in function
    def skewness(self):
        array = []
        for i in range(len(self.data)):
            array.append(stats.skew(np.array(self.data[i, :])))
        np.array(array).reshape((len(self.data), 1))
        return np.array(array)

    # feature 4. kurtosis values (calculate the flatness or peakness)
    def kurtosis(self):
        array = []
        for i in range(len(self.data)):
            array.append(stats.kurtosis(np.array(self.data[i, :])))
        np.array(array).reshape((len(self.data), 1))
        return np.array(array)

    # feature 6. chi-2 values (calculate the usefulness)
    def chisquare(self):
        array = []
        for i in range(len(self.data)):
            array.append(stats.chisquare(self.data, ddof=1))

        x = np.array(array)[:, 0, 0]  # cut the 3-dimension to 1 dimension for matrix concatenate
        return np.array(x)

def load_models():
    feature_m = joblib.load(open('featuresMatrix.pkl', 'rb'))
    k_means_m = joblib.load(open('kmeans_label.pkl', 'rb'))
    dbscan_m = joblib.load(open('dbscan_label.pkl', 'rb'))

    return feature_m, k_means_m, dbscan_m

def main():
    # MealData and NoMealData directory for the csv files
    meal_data = directory('./Test')
    # get the meal and nomeal data from separate folders
    meal = np.concatenate([getData(i) for i in meal_data])
    # transform them into dataframes
    meal_df = pd.DataFrame(meal)
    # trim the dataframe to 510 * 31 dataframe
    meal_df = meal_df[[i for i in range(30)]]
    # print(meal_df)
    # print(meal_amount_df)

    # Extract Features
    feat = features(meal_df.values)
    feat1 = feat.covariance()
    feat2 = feat.entropy()
    feat3 = feat.skewness()
    feat4 = feat.kurtosis()
    feat6 = feat.chisquare()

    # create a new matrix of features
    featuresMatrix = np.concatenate((feat1, feat2[:, None], feat3[:, None], feat4[:, None], feat6[:, None]), axis=1)

    # create a PCA with 5 components
    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(featuresMatrix)  # fit the model
    principalDf = pd.DataFrame(data=principalComponents)
    array = pca.explained_variance_ratio_  # get the variance ratio

    # loop through the variance ratio array and find the values that are greater than 0.2
    temp = []
    for i in range(len(array)):
        t = array[i]
        if t > 0.20:
            temp.append(i)

    principalDf = np.array(principalDf[[i for i in temp]])

    # load three models
    feature_model, kmeans_model, dbscan_model = load_models()

    # test using kmeans
    kmeans_KNN = KNeighborsClassifier(n_neighbors = 20)
    kmeans_KNN.fit(feature_model, kmeans_model)
    y_predict1 = np.asarray(kmeans_KNN.predict(principalDf))
    y_predict1 = y_predict1.transpose()

    # test using DBSCAN
    DBSCAN_KNN = KNeighborsClassifier(n_neighbors = 20)
    DBSCAN_KNN.fit(feature_model, dbscan_model)
    y_predict2 = np.asarray(DBSCAN_KNN.predict(principalDf))
    y_predict2 = y_predict2.transpose()

    # save into two columns
    temp = np.column_stack([y_predict1, y_predict2])

    np.savetxt('result.csv', temp, fmt = "%d", delimiter = ",")
    print("Result Successfully Saved!")

if __name__ == '__main__':
    main()
