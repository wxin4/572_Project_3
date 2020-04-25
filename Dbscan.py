from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import matplotlib.pyplot as plt
import k_means


def try_model_dbscan(label, label_class, idx_keep, bin_truth):
    # initialize raw clusters and bin clusters
    final_cluster = []
    final_idx = []
    result = []

    # loop thru the clusters and add the labels into cluster list and indices to index list
    for c in range(len(label_class)):
        cluster_c = []
        for i in range(len(label)):
            if label[i] == label_class[c]:
                cluster_c.append(i)

        final_index = []
        for i in cluster_c:
            final_index.append(idx_keep[i])

        final_cluster.append(cluster_c)
        final_idx.append(final_index)

    # put the ground truth labels into result label list
    for c in range(len(label_class)):
        result_label = [bin_truth[i] for i in final_cluster[c]]
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
    for c in range(len(label_class)):
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

    return bin_cluster, bin_index


def accuracy(matrix, bin_cluster, bin_index, ground_truth):
    idx_list = []

    # loop through every cluster and get the length of each of them
    for i in range(8):
        idx_list.append(len(bin_cluster[i]))

    # get the max index appeared in the clusters
    max_idx = idx_list.index(max(idx_list))
    feature_m = []
    for i in bin_cluster[max_idx]:
        feature_m.append(matrix[i])
    bin_index_1 = bin_index[max_idx]

    # call kmeans to cluster labels
    kmeans = KMeans(n_clusters = 60, random_state=0).fit(feature_m)
    label_1 = list(kmeans.labels_)

    # train the cluster and index by kmeans model
    density_cluster, density_index = k_means.kmeans_model(ground_truth, label_1, 60, bin_index_1)

    #     final_bin1 = bin_index[0]
    #     final_bin2 = bin_index[1]
    #     final_bin3 = bin_index[2]
    #     final_bin4 = bin_index[3]
    #     final_bin5 = bin_index[4]
    #     final_bin6 = bin_index[5]

    # initialize the 250 labels as 0s
    final_result = len(matrix) * [0]

    # bin indices that holds all the possible labels
    bin1 = density_index[0]
    bin2 = bin_index[1] + density_index[1]
    bin3 = bin_index[2] + density_index[2]
    bin4 = bin_index[3] + density_index[3]
    bin5 = bin_index[4] + density_index[4]
    bin6 = bin_index[5] + density_index[5]

    # create variables that hold how many 1,2,3,4,5,6s are there in each bin
    bin1_l = len(bin1) * [1]
    bin2_l = len(bin2) * [2]
    bin3_l = len(bin3) * [3]
    bin4_l = len(bin4) * [4]
    bin5_l = len(bin5) * [5]
    bin6_l = len(bin6) * [6]

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

    sse = kmeans.inertia_
    print("SSE value :", sse)

    return final_result


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


def dbscan_calculate(x):
    x = np.array(x)
    db = DBSCAN(eps=0.5, min_samples=3)
    db.fit(x)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    y_pred = db.fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred, cmap='viridis')
    plt.title("DBSCAN")
    plt.show()

    return n_clusters_