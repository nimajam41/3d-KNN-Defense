import numpy as np

class KNNDefense:

    def uniform_weighting(self, softmax):
        K = softmax.shape[0]

        weights = (np.ones(K) / K).reshape(K, 1)
        return (weights * softmax).sum(axis=0)
    

    def cbwe_weighting(self, softmax_vec):
        K = softmax_vec.shape[0]
        C = softmax_vec.shape[1]

        weights = np.log(C) + (np.log(softmax_vec) * softmax_vec).sum(1).reshape(K, 1)
        return (weights * softmax_vec).sum(axis=0)
    

    def cbwd_weighting(self, softmax_vec, M=20, P=3):
        K = softmax_vec.shape[0]
        sorted_softmax = np.sort(softmax_vec, 1)

        max_elems = sorted_softmax[:, -1:]
        top_M_elems = sorted_softmax[:, -1 * (M + 1):-1]

        weights = ((max_elems - top_M_elems) ** P).sum(1).reshape(K, 1)
        return (weights * softmax_vec).sum(axis=0)
    

    def knn(self, dists, softmax_vec, k, return_index):
        num_test, __ = dists.shape
        knn_softmax = []
        knn_neighbors = []

        for i in range(num_test):
            k_nearest = np.argsort(dists[i, :])[:k]
            if return_index:
                knn_neighbors += [k_nearest]

            k_nearest_softmax = softmax_vec[k_nearest]
            knn_softmax += [k_nearest_softmax]

        return np.array(knn_softmax), np.array(knn_neighbors)
     

    def compute_max_vote(self, labels):
        unq_labels, counts = np.unique(labels, return_counts=True)
        return unq_labels[np.argmax(counts)]


    def defense(self, dists, softmax_vec, labels=None, max_voting=False, k=40, 
                weight="CBWD", return_index=False):
        if max_voting:
            return_index = True

        knn_softmax, neighbors = self.knn(dists, softmax_vec, k, return_index)
        preds = []

        if not max_voting:
            if weight == "UNIFORM":
                weighting_func = self.uniform_weighting

            elif weight == "CBWE":
                weighting_func = self.cbwe_weighting

            elif weight == "CBWD":
                weighting_func = self.cbwd_weighting
                
            else:
                raise Exception("Invalid Weighting Function!")

            for i in range(knn_softmax.shape[0]):
                weighted_softmax = weighting_func(knn_softmax[i])
                preds += [np.argmax(weighted_softmax)]
        
        else:
            for i in range(neighbors.shape[0]):
                neighb_labels = labels[neighbors[i]]
                preds += [self.compute_max_vote(neighb_labels)]

        
        if return_index:
            return np.array(preds), neighbors
        
        else:
            return np.array(preds), None