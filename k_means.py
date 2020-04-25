import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# put clusters into kmeans model
from sklearn.cluster import KMeans


def kmeans_model(ground_truth, label, cluster_no, cluster_idx):
    # initialize raw clusters and bin clusters
    final_cluster = []
    final_idx = []
    result = []

    # loop thru the clusters and add the labels into cluster list and indices to index list
    for c in range(cluster_no):
        cluster_c = []
        for i in range(len(label)):
            if label[i] == c:
                cluster_c.append(i)

        final_index = []
        for i in cluster_c:
            final_index.append(cluster_idx[i])

        # add label clusters and indices clusters
        final_cluster.append(cluster_c)
        final_idx.append(final_index)

    # put the ground truth labels into result label list
    for c in range(cluster_no):
        result_label = [ground_truth[i] for i in final_cluster[c]]

        # get the maximum number of shown index
        result.append(max(set(result_label), key=result_label.count))

    # initialize calculated bin clusters and indices
    bin_cluster = []
    bin_index = []

    # initialize bin lists
    bin1 = []
    bin2 = []
    bin3 = []
    bin4 = []
    bin5 = []
    bin6 = []
    bin7 = []
    bin8 = []

    # initialize index lists
    index1 = []
    index2 = []
    index3 = []
    index4 = []
    index5 = []
    index6 = []
    index7 = []
    index8 = []

    # get each cluster and see which label it belongs to and distribute them into corresponding bin indices
    for c in range(cluster_no):
        if result[c] == 1:
            bin1 += final_cluster[c]
            index1 += final_idx[c]
        elif result[c] == 2:
            bin2 += final_cluster[c]
            index2 += final_idx[c]
        elif result[c] == 3:
            bin3 += final_cluster[c]
            index3 += final_idx[c]
        elif result[c] == 4:
            bin4 += final_cluster[c]
            index4 += final_idx[c]
        elif result[c] == 5:
            bin5 += final_cluster[c]
            index5 += final_idx[c]
        elif result[c] == 6:
            bin6 += final_cluster[c]
            index6 += final_idx[c]
        elif result[c] == 7:
            bin7 += final_cluster[c]
            index7 += final_idx[c]
        elif result[c] == 8:
            bin8 += final_cluster[c]
            index8 += final_idx[c]

    # add all the bin clusters to a final list
    bin_cluster.append(bin1)
    bin_cluster.append(bin2)
    bin_cluster.append(bin3)
    bin_cluster.append(bin4)
    bin_cluster.append(bin5)
    bin_cluster.append(bin6)
    bin_cluster.append(bin7)
    bin_cluster.append(bin8)

    # add all the indices of bin to a final list
    bin_index.append(index1)
    bin_index.append(index2)
    bin_index.append(index3)
    bin_index.append(index4)
    bin_index.append(index5)
    bin_index.append(index6)
    bin_index.append(index7)
    bin_index.append(index8)
    # print(bin_index)

    return bin_cluster, bin_index

# go thru each label and see if the test and pred have the same label
def knn_score(pred, test):
    score = 0
    result = 0
    for i in range(len(pred)):
        if pred[i] == test[i]:
            score = score + 1

    # compare each value and calculate the correctness score
    result = score / (len(pred))
    print("Accuracy Score :", result)


def accuracy(matrix, bin_index):

    # initialize the 250 labels as 0s
    final_result = len(matrix) * [0]

    # bin indices that holds all the possible labels
    bin1 = bin_index[0]
    bin2 = bin_index[1]
    bin3 = bin_index[2]
    bin4 = bin_index[3]
    bin5 = bin_index[4]
    bin6 = bin_index[5]

    # create variables that hold how many 1,2,3,4,5,6s are there in each bin
    bin1_l = len(bin1)*[1]
    bin2_l = len(bin2)*[2]
    bin3_l = len(bin3)*[3]
    bin4_l = len(bin4)*[4]
    bin5_l = len(bin5)*[5]
    bin6_l = len(bin6)*[6]

    # print(len(final_bin1))
    # print(len(final_bin2))
    # print(len(final_bin3))
    # print(len(final_bin4))
    # print(len(final_bin5))
    # print(len(final_bin6))

    # replace labels in final result by each of the bins
    for i in bin1:
        final_result[i] = bin1_l.pop()
    for i in bin2:
        final_result[i] = bin2_l.pop()
    for i in bin3:
        final_result[i] = bin3_l.pop()
    for i in bin4:
        final_result[i] = bin4_l.pop()
    for i in bin5:
        final_result[i] = bin5_l.pop()
    for i in bin6:
        final_result[i] = bin6_l.pop()

    # print(final_result)

    return final_result


def fit(x):
    score = 0
    k = 0
    x = np.array(x)
    sse = {}
    for k in range(1, 9):
        x = pd.DataFrame(x)
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(x)
        x["clusters"] = kmeans.labels_
        # print(data["clusters"])
        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
        score += sse[k]

    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()

    return sse[k] / k


def show_clusters(x):
    x = np.array(x)
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(x)
    pred_kmeans = kmeans.predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=pred_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

    return pred_kmeans