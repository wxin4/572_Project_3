import os

import joblib
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

import k_means
import Dbscan
from Dbscan import dbscan_calculate
from purity import purity_calculation
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import *
import pandas as pd
from sklearn import svm, metrics
from scipy import stats
from scipy.stats import entropy
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# traverse to the directory
from ground_truth import extract_ground_truth


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


# KFold function for KMeans
def K_Fold_Kmeans(matrix):
    print('K-Fold for Kmeans:')
    ground_truth = extract_ground_truth()
    ground_truth = np.asarray(ground_truth)
    index = 0

    # apply kfold by spliting 10 times of the whole training set
    k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
    k_fold.get_n_splits(matrix)
    final_idx = []

    # split the matrix and get training set and testing set in every folding
    for idx_train, idx_test in k_fold.split(matrix):
        X_train, X_test = matrix[idx_train], matrix[idx_test]

        # Y train and test using the last column
        Y_train, Y_test = ground_truth[idx_train], ground_truth[idx_test]

        # get the kmeans
        kmeans = KMeans(n_clusters = 85, random_state=0).fit(X_train)
        print('k =', index + 1)
        print('SSE value:', kmeans.inertia_)
        label = list(kmeans.labels_)

        # save all the filtered label indices into final index list and call dbscan model
        for i in range(len(label)):
            final_idx.append(i)
        bin_cluster, bin_index = k_means.kmeans_model(Y_train, label, 85, final_idx)

        # save all the filtered label into final label list and call accuracy report
        l = k_means.accuracy(X_train, bin_index)

        # KNN score
        knn = KNeighborsClassifier(n_neighbors = 20)
        knn.fit(X_train, l)
        Y_pred = knn.predict(X_test)
        k_means.knn_score(Y_pred, Y_test)
        index += 1
        print()

        # apply SVM and fit the model with training sets
        model = svm.SVC(kernel='rbf')
        model.fit(X_train, Y_train)

    # Train clusters with full dataset
    print("KMeans ALL DATA (because SSE value is the lowest & Accu is the highest :")
    kmeans = KMeans(n_clusters = 80, random_state = 0).fit(matrix)
    print("SSE value of the clusters: ", kmeans.inertia_)
    label = list(kmeans.labels_)

    # save all the filtered label indices into final index list and call dbscan model
    for i in range(len(label)):
        final_idx.append(i)
    bin_cluster, bin_index = k_means.kmeans_model(ground_truth, label, 80, final_idx)

    # save all the filtered label into final label list and call accuracy report
    l = k_means.accuracy(matrix, bin_index)

    # dump the trained label into pkl file
    modelpkl(l, 'kmeans_label.pkl')

    # show the SSE value graph and the cluster graph
    k_means.fit(matrix)
    k_means.show_clusters(matrix)


# KFold function for DBSCAN
def K_Fold_DBSCAN(matrix):
    print()
    print('K-Fold for DBSCAN:')
    print()
    ground_truth = extract_ground_truth()
    ground_truth = np.asarray(ground_truth)

    # call KFold function to split the data to 5 clusters
    kf = KFold(n_splits=5)
    kf.get_n_splits(matrix)

    index = 0
    final_idx = []

    # split the matrix and get training set and testing set in every folding
    for idx_train, idx_test in kf.split(matrix):
        # Y train and test using the last column
        X_train, X_test = matrix[idx_train], matrix[idx_test]
        Y_train, Y_test = ground_truth[idx_train], ground_truth[idx_test]

        # dbscan we use 5 min samples because of the Accuracy score
        dbscan = DBSCAN(eps = 0.5, min_samples = 5).fit(X_train)
        label = dbscan.labels_
        label_idx = np.unique(label)
        label = list(label)

        # save all the filtered label indices into final index list and call dbscan model
        for i in range(len(label)):
            final_idx.append(i)
        bin_cluster, bin_index = Dbscan.try_model_dbscan(label, label_idx, final_idx, Y_train)

        # save all the filtered label into final label list and call accuracy report
        l = Dbscan.accuracy(X_train, bin_cluster, bin_index, Y_train)

        # get the KNN score
        knn = KNeighborsClassifier(n_neighbors = 20)
        knn.fit(X_train, l)
        y_predict = knn.predict(X_test)
        Dbscan.knn_score(y_predict, Y_test)
        index += 1
        print()

    # training all the data to DBSCAN model
    print("DBSCAN All data (because SSE value is lowest & Accu score is highest) :")
    dbscan = DBSCAN(eps = 0.5, min_samples = 5).fit(matrix)
    label = dbscan.labels_
    label_idx = np.unique(label)
    label = list(label)

    # save all the filtered label indices into final index list and call dbscan model
    for i in range(len(label)):
        final_idx.append(i)
    bin_cluster, bin_index = Dbscan.try_model_dbscan(label, label_idx, final_idx, ground_truth)

    # save all the filtered label into final label list and call accuracy report
    l = Dbscan.accuracy(matrix, bin_cluster, bin_index, ground_truth)

    # dump the trained label into pkl file
    modelpkl(l, 'dbscan_label.pkl')

    # show the clusters graph
    Dbscan.dbscan_calculate(matrix)


def modelpkl(model, name='ModelForTesting.pkl'):
    if name == 'kmeans_label.pkl':
        joblib.dump(model, 'kmeans_label.pkl')
    elif name == 'dbscan_label.pkl':
        joblib.dump(model, 'dbscan_label.pkl')
    else:
        joblib.dump(model, 'ModelForTesting.pkl')

    print()
    print('Model Successfully Built!')


def main():
    # MealData and NoMealData directory for the csv files
    meal_data = directory('./MealData')
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

    # call KFold method to kmeans and DBSCAN
    K_Fold_Kmeans(principalDf)
    K_Fold_DBSCAN(principalDf)

    # save featuresMatrix into pkl file
    with open('featuresMatrix.pkl', 'wb') as f:
        joblib.dump(principalDf, f)

    # apply kfold using the pca dataframe and the label
    # K_Fold(np.concatenate((principalDf[[i for i in temp]], label[:, None]), axis=1))
    #
    # # Y_test data and train the model with SVM and put in to pkl file
    # Y_t = np.concatenate((principalDf[[i for i in temp]], label[:, None]), axis=1)[:, -1]
    # model = svm.SVC(kernel='rbf')
    # model.fit(principalDf[[i for i in temp]], Y_t)
    # modelpkl(model)

if __name__ == '__main__':
    main()
