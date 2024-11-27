import os
import copy
import time
import pickle
import argparse
import numpy as np
# from pmle_151 import pMLEGMM
from pmle import pMLEGMM
from CTDGMR.utils import *
from CTDGMR.minCTD import *
from CTDGMR.distance import GMM_CTD, GMM_L2
from tqdm import tqdm


def estimate_Gaussian(data):
    n, d = data.shape
    mean = data.mean(0)
    diff = data - mean
    cov = np.dot(diff.T, diff) / n + 1e-4 * np.eye(d)
    return mean, cov


def robustmedian(
    subset_means,
    subset_covs,
    subset_weights,
    ground_distance="L2",
    coverage_ratio=0.5,
):
    n_split = len(subset_means)
    pairwisedist = np.zeros((n_split, n_split))
    for i in range(n_split):
        for j in range(n_split):
            if "CTD" in ground_distance:
                pairwisedist[i, j] = GMM_CTD(
                    [subset_means[i], subset_means[j]],
                    [subset_covs[i], subset_covs[j]],
                    [subset_weights[i], subset_weights[j]],
                    ground_distance=ground_distance.split("-")[1],
                    matrix=False)
            else:
                pairwisedist[i, j] = np.sqrt(
                    GMM_L2(
                        [subset_means[i], subset_means[j]],
                        [subset_covs[i], subset_covs[j]],
                        [subset_weights[i], subset_weights[j]],
                    ))
    which_GMM = np.argmin(np.quantile(pairwisedist, q=coverage_ratio, axis=1))
    output = [which_GMM, pairwisedist]

    return output


def ared_threshold(distance_to_center, pairwisedist, which_GMM, m):
    half_of_num_machine = int(distance_to_center.shape[0] // 2)
    indices = np.argsort(distance_to_center)
    sorted_distance = distance_to_center[indices]
    c_sd = np.sort(pairwisedist[which_GMM])[half_of_num_machine]
    threshold = np.sqrt(np.log(m / 2) * np.log(np.log(m)) * 2)
    if (np.sum(sorted_distance > threshold * c_sd) == 0):
        truncation = distance_to_center.shape[0]
    else:
        truncation = np.where(sorted_distance > threshold * c_sd)[0][0]

    return indices[:truncation]


def ccred(means, covs, weights, means_init, covs_init, weights_init):
    K, D = means_init.shape
    n_split = means.shape[0] // K
    cost_matrix = GMM_CTD(
        means=[means, means_init],
        covs=[covs, covs_init],
        weights=[weights / n_split, weights_init],
        ground_distance="KL",
        matrix=True,
    )
    clustering_matrix = (cost_matrix.T == np.min(cost_matrix, 1)).T
    ccred_means, ccred_covs, ccred_weights = np.zeros((K, D)), np.zeros(
        (K, D, D)), np.zeros(K)
    r_coats = 0
    indice_lengths = []

    for k in range(K):
        group_means = means[clustering_matrix[:, k]]
        group_covs = covs[clustering_matrix[:, k]]
        local_closest_indices = np.argsort(cost_matrix[clustering_matrix[:, k],
                                                       k])[:int(n_split // 2)]
        r_coat = cost_matrix[clustering_matrix[:, k],
                             k][local_closest_indices[-1]]
        r_coats += r_coat
        # print(k, local_closest_indices)
        # indice_lengths.append(local_closest_indices.shape[0])
        indice_lengths.append(local_closest_indices)
        ccred_means[k], ccred_covs[k] = barycenter(
            group_means[local_closest_indices],
            group_covs[local_closest_indices],
            weights[clustering_matrix[:, k]][local_closest_indices],
            mean_init=means_init[k],
            cov_init=means_init[k],
            ground_distance="KL")
        ccred_weights[k] = np.sum(
            weights[clustering_matrix[:, k]][local_closest_indices])
    ccred_weights /= ccred_weights.sum()

    return ccred_means, ccred_covs, ccred_weights, r_coats, indice_lengths


def cared(means, covs, weights, means_init, covs_init, weights_init):
    K, D = means_init.shape
    n_split = means.shape[0] / K
    cost_matrix = GMM_CTD(
        means=[means, means_init],
        covs=[covs, covs_init],
        weights=[weights / n_split, weights_init],
        ground_distance="KL",
        matrix=True,
    )
    clustering_matrix = (cost_matrix.T == np.min(cost_matrix, 1)).T
    cared_means, cared_covs, cared_weights = np.zeros((K, D)), np.zeros(
        (K, D, D)), np.zeros(K)

    for k in range(K):
        group_means = means[clustering_matrix[:, k]]
        group_covs = covs[clustering_matrix[:, k]]
        local_closest_indices = np.argsort(cost_matrix[clustering_matrix[:, k],
                                                       k])[:int(n_split // 2)]
        r_coat = cost_matrix[clustering_matrix[:, k],
                             k][local_closest_indices[-1]]
        a_coat = r_coat * np.sqrt(
            np.log(n_split / 2) * np.log(np.log(n_split)) * 2)
        local_closest_indices = np.where(cost_matrix[clustering_matrix[:, k],
                                                     k] <= a_coat)
        # only aggregate 50% for each component
        cared_means[k], cared_covs[k] = barycenter(
            group_means[local_closest_indices],
            group_covs[local_closest_indices],
            weights[clustering_matrix[:, k]][local_closest_indices],
            mean_init=means_init[k],
            cov_init=means_init[k],
            ground_distance="KL")
        cared_weights[k] = np.sum(
            weights[clustering_matrix[:, k]][local_closest_indices])
    cared_weights /= cared_weights.sum()

    return cared_means, cared_covs, cared_weights


# -------------------------------------------------------------------
# split sample to different machines and fit mixture on each machine
# -------------------------------------------------------------------
def main(random_state, local_ss, failure_type):

    n_split = 30
    K, D = 10, 50
    np.random.seed(random_state)
    local_per_class = int(local_ss // n_split)

    local_means = [np.empty((K, D)) for _ in range(n_split)]
    local_covs = [np.empty((K, D, D)) for _ in range(n_split)]
    local_weights = [np.empty((K, )) for _ in range(n_split)]
    local_ARI = [None] * n_split
    local_ll = [None] * n_split

    # load preprcoessed dataset
    digits_train = np.load(
        "preprocessed_data/NIST_feature_digits_train_50d.npy")
    digits_train_label = np.load(
        "preprocessed_data/NIST_label_digits_train.npy")
    digits_test = np.load("preprocessed_data/NIST_feature_digits_test_50d.npy")
    digits_test_label = np.load("preprocessed_data/NIST_label_digits_test.npy")
    letters_train = np.load(
        "preprocessed_data/NIST_feature_letters_train_50d.npy")
    # letters_train_label = np.load(
    #     "preprocessed_data/NIST_label_letters_train.npy")

    print(digits_train.shape, digits_test.shape, letters_train.shape)

    shuffled_index = np.random.permutation(np.arange(digits_train.shape[0]))
    digits_train = digits_train[shuffled_index]

    for split in tqdm(range(n_split)):
        local = []
        for i in range(10):
            local.append(digits_train[digits_train_label == i][(
                split * local_per_class):((split + 1) * local_per_class)])
        local = np.vstack(local)
        # random warm start first and then start from the true initial value
        gmmk = pMLEGMM(
            n_components=K,
            cov_reg=1.0 / np.sqrt(local.shape[0]),
            # cov_reg=1.0 / local.shape[0],
            covariance_type="full",
            max_iter=50,
            n_init=10,
            tol=1e-10,
            random_state=10,
            verbose=0,
            verbose_interval=1,
            # init_params="k-means++",
            warm_start=True,
        )
        start_time = time.time()
        gmmk.fit(local)
        gmmk.max_iter = 10000
        gmmk.fit(local)

        local_pmle_means, local_pmle_covs, local_pmle_weights = (
            gmmk.means_,
            gmmk.covariances_,
            gmmk.weights_,
        )
        local_means[split] = local_pmle_means
        local_covs[split] = local_pmle_covs
        local_weights[split] = local_pmle_weights
        local_resp, local_predicted_label = label_predict(
            local_pmle_weights,
            local_pmle_means,
            local_pmle_covs,
            digits_test,
            return_resp=True,
        )
        local_ARI[split] = ARI(digits_test_label, local_predicted_label)
        local_ll[split] = np.log(local_resp.sum(1)).sum(0)

    # save output data

    output_data = {
        "local_ARI": local_ARI,
        "local": (local_means, local_covs, local_weights),
        "local_ll": local_ll,
    }

    save_folder = "Local"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    save_file = os.path.join(
        save_folder,
        "case_" + str(random_state) + "_nsplit_" + str(n_split) + "_ncomp_" +
        str(K) + "_d_" + str(D) + "_ss_" + str(local_ss) + ".pickle",
    )

    f = open(save_file, "wb")
    pickle.dump(output_data, f)
    f.close()

    with open(save_file, 'rb') as f:
        output_data = pickle.load(f)
    local_means, local_covs, local_weights = output_data['local']
    local_ARI = output_data["local_ARI"]
    local_ll = output_data["local_ll"]
    # -------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------

    # for failure_rate in [0.1]:
    for failure_rate in tqdm([0.0, 0.1, 0.2, 0.3, 0.4]):
        if failure_rate != 0:
            # Load local estimates
            save_folder = 'Local'
            save_file = os.path.join(
                save_folder,
                "case_" + str(random_state) + "_nsplit_" + str(n_split) +
                "_ncomp_" + str(K) + "_d_" + str(D) + "_ss_" + str(local_ss) +
                ".pickle",
            )

            with open(save_file, 'rb') as f:
                output_data = pickle.load(f)
            local_means, local_covs, local_weights = output_data['local']
            local_ARI = output_data["local_ARI"]
            local_ll = output_data["local_ll"]

            #------------------------------------
            # generate Byzantine failure
            #------------------------------------
            if failure_type == "machine":
                rng = np.random.default_rng(random_state)
                byzantine_machine_index = rng.choice(n_split,
                                                     int(n_split *
                                                         failure_rate),
                                                     replace=False)
                for b_index in byzantine_machine_index:
                    local = letters_train[rng.choice(letters_train.shape[0],
                                                     local_ss,
                                                     replace=False)]
                failure_indices = []
                for k in range(K):
                    failure_indices.append(byzantine_machine_index)
                gmmk = pMLEGMM(
                    n_components=K,
                    cov_reg=1.0 / np.sqrt(local.shape[0]),
                    covariance_type="full",
                    max_iter=50,
                    n_init=10,
                    tol=1e-10,
                    random_state=10,
                    verbose=0,
                    verbose_interval=1,
                    warm_start=True,
                )
                start_time = time.time()
                gmmk.fit(local)
                gmmk.max_iter = 10000
                gmmk.fit(local)

                local_pmle_means, local_pmle_covs, local_pmle_weights = (
                    gmmk.means_,
                    gmmk.covariances_,
                    gmmk.weights_,
                )
                local_means[b_index] = local_pmle_means
                local_covs[b_index] = local_pmle_covs
                local_weights[b_index] = local_pmle_weights
                local_resp, local_predicted_label = label_predict(
                    local_pmle_weights,
                    local_pmle_means,
                    local_pmle_covs,
                    digits_test,
                    return_resp=True,
                )
                local_ARI[b_index] = ARI(digits_test_label,
                                         local_predicted_label)
                local_ll[b_index] = np.log(local_resp.sum(1)).sum(0)
                # local = []
                # for i in range(10):
                #     ith_letter = letters_train[letters_train_label == i]

                #     local.append(ith_letter[np.random.choice(
                #         ith_letter.shape[0], local_per_class, replace=False)])
                # local.append(letters_train[letters_train_label == i][(
                #     b_index * local_per_class):((b_index + 1) *
                #                                 local_per_class)])
                # local = np.vstack(local)
                # local = letters_train[letters_train_label == b_index][:local_ss]
                # random warm start first and then start from the true initial value
            elif failure_type == "component":
                contaminated_label = [np.zeros(K) for _ in range(n_split)]
                per_comp_failure_num = int(np.floor(failure_rate * n_split))
                rng = np.random.default_rng(random_state)
                failure_indices = []
                for k in range(K):
                    # randomly select the Byzantine failure components
                    # The last machine is always Byzantine failure free
                    failure_index = rng.choice(n_split - 1,
                                               per_comp_failure_num,
                                               replace=False)
                    # print(k, failure_index)
                    failure_indices.append(failure_index)
                    # generate attacks
                    for idx in failure_index:
                        # local_data = letters_train[letters_train_label == k]
                        local_mean, local_cov = estimate_Gaussian(
                            letters_train[rng.choice(letters_train.shape[0],
                                                     100,
                                                     replace=False)])
                        # local_data[rng.choice(local_data.shape[0],
                        #                       int(local_weights[idx][k] *
                        #                           local_ss),
                        #                       replace=False)])
                        local_means[idx][k] = copy.deepcopy(local_mean)
                        local_covs[idx][k] = copy.deepcopy(local_cov)
                        contaminated_label[idx][k] = 1
                for i in range(n_split):
                    local_resp, local_predicted_label = label_predict(
                        local_weights[i],
                        local_means[i],
                        local_covs[i],
                        digits_test,
                        return_resp=True,
                    )
                    local_ARI[i] = ARI(digits_test_label,
                                       local_predicted_label)
                    local_ll[i] = np.log(local_resp.sum(1)).sum(0)

            else:
                raise ValueError("This failure type is not implemented!")

            save_file = os.path.join(
                save_folder,
                "case_" + str(random_state) + "_nsplit_" + str(n_split) +
                "_ncomp_" + str(K) + "_d_" + str(D) + "_ss_" + str(local_ss) +
                "_failuretate_" + str(failure_rate) + "_failuretype_" +
                str(failure_type) + ".pickle",
            )

            output_data = {
                "local_ARI": local_ARI,
                "local": (local_means, local_covs, local_weights),
                "local_ll": local_ll,
            }
            f = open(save_file, "wb")
            pickle.dump(output_data, f)
            f.close()

        else:
            byzantine_machine_index = []
            contaminated_label = []
            failure_indices = [[] for _ in range(n_split)]

        output_data = {}
        if failure_type == "machine":
            output_data["byzantine_machine"] = byzantine_machine_index
        elif failure_type == "component":
            output_data["contaminated_label"] = contaminated_label

        # mycost = GMM_CTD(
        #     means=[np.concatenate(local_means),
        #            np.concatenate(local_means)],
        #     covs=[np.concatenate(local_covs),
        #           np.concatenate(local_covs)],
        #     weights=[
        #         np.concatenate(local_weights) / n_split,
        #         np.concatenate(local_weights) / n_split
        #     ],
        #     ground_distance="KL",
        #     matrix=True,
        # )
        # ------------------------------
        # COAT
        # ------------------------------
        start_time = time.time()
        which_GMM, pairwisedist = robustmedian(
            local_means,
            local_covs,
            local_weights,
            ground_distance="CTD-KL",
            coverage_ratio=0.5,
        )
        coat_time = time.time() - start_time

        output_data["coat_coat"] = coat_time
        output_data["coat_index"] = which_GMM
        output_data["coat_ARI"] = local_ARI[which_GMM]
        output_data["coat_ll"] = local_ll[which_GMM]

        coat_means, coat_covs, coat_weights = (
            local_means[which_GMM],
            local_covs[which_GMM],
            local_weights[which_GMM],
        )

        # ------------------------------
        # CRED
        # ------------------------------
        start_time = time.time()
        closest_indices = np.argsort(pairwisedist[which_GMM])[:int(n_split //
                                                                   2)]
        reduced_gmm = GMR_CTD(
            np.concatenate([local_means[index] for index in closest_indices]),
            np.concatenate([local_covs[index] for index in closest_indices]),
            np.concatenate([local_weights[index]
                            for index in closest_indices]) /
            closest_indices.shape[0],
            K,
            ground_distance="KL",
            init_method="user",
            means_init=coat_means,
            covs_init=coat_covs,
            weights_init=coat_weights,
        )
        reduced_gmm.iterative()
        cred_time = time.time() - start_time

        cred_means, cred_covs, cred_weights = (
            reduced_gmm.reduced_means,
            reduced_gmm.reduced_covs,
            reduced_gmm.reduced_weights,
        )

        cred_resp, cred_predicted_label = label_predict(
            cred_weights,
            cred_means,
            cred_covs,
            digits_test,
            return_resp=True,
        )

        cred_ARI = ARI(digits_test_label, cred_predicted_label)
        cred_ll = np.log(cred_resp.sum(1)).sum(0)

        output_data["cred_time"] = cred_time
        output_data["cred_ARI"] = cred_ARI
        output_data["cred"] = (cred_means, cred_covs, cred_weights)
        output_data["cred_ll"] = cred_ll

        # ------------------------------
        # ARED
        # ------------------------------
        start_time = time.time()
        distance_to_center = pairwisedist[which_GMM]
        ared_untruncated_indices = ared_threshold(distance_to_center,
                                                  pairwisedist, which_GMM,
                                                  n_split)
        model = GMR_CTD(
            np.concatenate(
                [local_means[index] for index in ared_untruncated_indices]),
            np.concatenate(
                [local_covs[index] for index in ared_untruncated_indices]),
            np.concatenate(
                [local_weights[index] for index in ared_untruncated_indices]) /
            ared_untruncated_indices.shape[0],
            K,
            ground_distance="KL",
            init_method="user",
            means_init=coat_means,
            covs_init=coat_covs,
            weights_init=coat_weights,
        )
        model.iterative()
        ared_time = time.time() - start_time

        ared_means, ared_covs, ared_weights = (
            model.reduced_means,
            model.reduced_covs,
            model.reduced_weights,
        )
        ared_resp, ared_predicted_label = label_predict(
            ared_weights,
            ared_means,
            ared_covs,
            digits_test,
            return_resp=True,
        )

        ared_ARI = ARI(digits_test_label, ared_predicted_label)
        ared_ll = np.log(ared_resp.sum(1)).sum(0)

        output_data["ared_time"] = ared_time
        output_data["ared_ARI"] = ared_ARI
        output_data["ared"] = (
            ared_means,
            ared_covs,
            ared_weights,
        )
        output_data["ared_ll"] = ared_ll
        output_data["ared_trimmed"] = ared_untruncated_indices

        # ------------------------------
        # CCRED
        # ------------------------------
        start_time = time.time()
        optimal_rcoat = np.Inf

        for i in range(n_split):
            temp_means, temp_covs, temp_weights, r_coats, single_cluster = ccred(
                np.concatenate(local_means),
                np.concatenate(local_covs),
                np.concatenate(local_weights),
                local_means[i],
                local_covs[i],
                local_weights[i],
            )
            if r_coats < optimal_rcoat and (1 not in [
                    i.shape[0] for i in single_cluster
            ]):
                optimal_rcoat = r_coats
                ccred_means = temp_means
                ccred_covs = temp_covs
                ccred_weights = temp_weights
                optimal_rcoat_machine = i
        ccred_time = time.time() - start_time

        # print(optimal_rcoat_machine, single_cluster)

        ccred_resp, ccred_predicted_label = label_predict(
            ccred_weights,
            ccred_means,
            ccred_covs,
            digits_test,
            return_resp=True,
        )

        ccred_ARI = ARI(digits_test_label, ccred_predicted_label)
        ccred_ll = np.log(ccred_resp.sum(1)).sum(0)

        output_data["ccred_time"] = ccred_time
        output_data["ccred_ARI"] = ccred_ARI
        output_data["ccred"] = (ccred_means, ccred_covs, ccred_weights)
        output_data["ccred_ll"] = ccred_ll

        # ------------------------------
        # CARED
        # ------------------------------
        start_time = time.time()
        cared_means, cared_covs, cared_weights = cared(
            np.concatenate(local_means),
            np.concatenate(local_covs),
            np.concatenate(local_weights),
            local_means[optimal_rcoat_machine],
            local_covs[optimal_rcoat_machine],
            local_weights[optimal_rcoat_machine],
        )
        cared_time = time.time() - start_time

        cared_resp, cared_predicted_label = label_predict(
            cared_weights,
            cared_means,
            cared_covs,
            digits_test,
            return_resp=True,
        )

        cared_ARI = ARI(digits_test_label, cared_predicted_label)
        cared_ll = np.log(cared_resp.sum(1)).sum(0)

        output_data["cred_time"] = cared_time
        output_data["cared_ARI"] = cared_ARI
        output_data["cared"] = (
            cared_means,
            cared_covs,
            cared_weights,
        )
        output_data["cared_ll"] = cared_ll
        # ------------------------------
        # GMR
        # ------------------------------
        start_time = time.time()
        reduced_gmm = GMR_CTD(
            np.concatenate(local_means),
            np.concatenate(local_covs),
            np.concatenate(local_weights) / n_split,
            K,
            ground_distance="KL",
            init_method="user",
            means_init=coat_means,
            covs_init=coat_covs,
            weights_init=coat_weights,
        )
        reduced_gmm.iterative()
        gmr_time = time.time() - start_time

        gmr_means, gmr_covs, gmr_weights = (
            reduced_gmm.reduced_means,
            reduced_gmm.reduced_covs,
            reduced_gmm.reduced_weights,
        )
        gmr_resp, gmr_predicted_label = label_predict(gmr_weights,
                                                      gmr_means,
                                                      gmr_covs,
                                                      digits_test,
                                                      return_resp=True)

        gmr_ARI = ARI(digits_test_label, gmr_predicted_label)
        gmr_ll = np.log(gmr_resp.sum(1)).sum(0)

        output_data["gmr_time"] = gmr_time
        output_data["gmr_ARI"] = gmr_ARI
        output_data["gmr"] = (gmr_means, gmr_covs, gmr_weights)
        output_data["gmr_ll"] = gmr_ll

        # ------------------------------
        # Trimmed k-barycenter with the 50% as the trimming level
        # ------------------------------
        current_time = time.time()
        model = GMR_PCTD(
            np.concatenate(local_means),
            np.concatenate(local_covs),
            np.concatenate(local_weights) / n_split,
            K,
            ground_distance="KL",
            init_method="user",
            alpha=0.5,  # 50% trimming
            means_init=coat_means,
            covs_init=coat_covs,
            weights_init=coat_weights,
        )

        model.iterative()
        trim_time = time.time() - current_time
        trim_means, trim_covs, trim_weights = (
            model.reduced_means,
            model.reduced_covs,
            model.reduced_weights,
        )
        trim_resp, trim_predicted_label = label_predict(trim_weights,
                                                        trim_means,
                                                        trim_covs,
                                                        digits_test,
                                                        return_resp=True)
        trim_ARI = ARI(digits_test_label, trim_predicted_label)
        trim_ll = np.log(trim_resp.sum(1)).sum(0)

        output_data["trim_time"] = trim_time
        output_data["trim_ARI"] = trim_ARI
        output_data["trim"] = (
            trim_means,
            trim_covs,
            trim_weights,
        )
        output_data["trim_ll"] = trim_ll
        output_data["trim_label"] = model.trimmed_label

        # ------------------------------
        # GMR + oracle weights
        # ------------------------------
        oracle_trimmed_weights = np.array([
            local_weights[i][j] for i in range(n_split) for j in range(K)
            if i not in failure_indices[j]
        ])
        oracle_trimmed_weights /= oracle_trimmed_weights.sum()

        oracle = GMR_CTD(
            np.stack([
                local_means[i][j] for i in range(n_split) for j in range(K)
                if i not in failure_indices[j]
            ]),
            np.stack([
                local_covs[i][j] for i in range(n_split) for j in range(K)
                if i not in failure_indices[j]
            ]),
            oracle_trimmed_weights,
            K,
            ground_distance="KL",
            init_method="user",
            means_init=coat_means,
            covs_init=coat_covs,
            weights_init=coat_weights,
        )
        start_time = time.time()
        oracle.iterative()
        oracle_time = time.time() - start_time
        oracle_means, oracle_covs, oracle_weights = (
            oracle.reduced_means,
            oracle.reduced_covs,
            oracle.reduced_weights,
        )
        oracle_resp, oracle_predicted_label = label_predict(
            oracle_weights,
            oracle_means,
            oracle_covs,
            digits_test,
            return_resp=True,
        )

        oracle_ARI = ARI(digits_test_label, oracle_predicted_label)
        oracle_ll = np.log(oracle_resp.sum(1)).sum(0)

        output_data["oracle_time"] = oracle_time
        output_data["oracle_ARI"] = oracle_ARI
        output_data["oracle"] = (
            oracle_means,
            oracle_covs,
            oracle_weights,
        )
        output_data["oracle_ll"] = oracle_ll

        for key, item in output_data.items():
            # if "ll" in key and "local" not in key or "ARI" in key and "local" not in key:
            if "ARI" in key and "local" not in key:
                # if "ARI" in key:
                # if ("ll" in key or "W1" in key) and "local" not in key:
                print(key, item)

        save_folder = "aggregation"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        save_file = os.path.join(
            save_folder,
            "case_" + str(random_state) + "_nsplit_" + str(n_split) +
            "_ncomp_" + str(K) + "_d_" + str(D) + "_ss_" + str(local_ss) +
            "_failurerate_" + str(failure_rate) + "_failuretype_" +
            str(failure_type) + ".pickle",
        )

        f = open(save_file, "wb")
        pickle.dump(output_data, f)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset split GMM estimator comparison")
    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="index of repetition")
    parser.add_argument("--local_ss",
                        type=int,
                        default=30000,
                        help="Total sample size from a GMM")
    parser.add_argument("--failure_type",
                        type=str,
                        default='machine',
                        help="Failure type: machine or component")

    args = parser.parse_args()
    local_ss = int(args.local_ss)
    seed = args.seed
    failure_type = args.failure_type

    main(seed, local_ss, failure_type)
