import os
import copy
import time
import pickle
import argparse
import numpy as np
from pmle import pMLEGMM
from CTDGMR.utils import *
from CTDGMR.minCTD import *
from CTDGMR.distance import GMM_CTD, GMM_L2
from tqdm import tqdm


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
                    matrix=False,
                )
            else:
                pairwisedist[i, j] = GMM_L2(
                    [subset_means[i], subset_means[j]],
                    [subset_covs[i], subset_covs[j]],
                    [subset_weights[i], subset_weights[j]],
                )
    which_GMM = np.argmin(np.quantile(pairwisedist, q=coverage_ratio, axis=1))
    output = [which_GMM, pairwisedist]

    return output


# def ared_threshold(distance_to_center, pairwisedist, N, option=1):
#     half_of_num_machine = int(distance_to_center.shape[0] // 2)
#     indices = np.argsort(distance_to_center)
#     sorted_distance = distance_to_center[indices]
#     closest_indices = indices[:half_of_num_machine]

#     if option == 1:
#         c_sd = np.sum(
#             distance_to_center[closest_indices]) / (2 * half_of_num_machine)
#         threshold = 3
#     elif option == 2:
#         c_sd = np.sum(
#             distance_to_center[closest_indices]) / (2 * half_of_num_machine)
#         threshold = np.log(np.log(N))
#     elif option == 3:
#         rows, cols = np.ix_(closest_indices, closest_indices)
#         c_sd = np.sum(pairwisedist[rows, cols]) / (2 * half_of_num_machine *
#                                                    (half_of_num_machine - 1))
#         threshold = 3
#     elif option == 4:
#         rows, cols = np.ix_(closest_indices, closest_indices)
#         c_sd = np.sum(pairwisedist[rows, cols]) / (2 * half_of_num_machine *
#                                                    (half_of_num_machine - 1))
#         threshold = np.log(np.log(N))
#     if np.sum(sorted_distance > threshold * c_sd) == 0:
#         truncation = distance_to_center.shape[0]
#     else:
#         truncation = np.where(sorted_distance > threshold * c_sd)[0][0]

#     return indices[:truncation]


def ared_threshold(distance_to_center, pairwisedist, which_GMM):
    half_of_num_machine = int(distance_to_center.shape[0] // 2)
    indices = np.argsort(distance_to_center)
    sorted_distance = distance_to_center[indices]

    threshold = np.log(distance_to_center.shape[0] / 2) * np.log(
        np.log(distance_to_center.shape[0]))
    c_sd = np.sort(pairwisedist[which_GMM])[half_of_num_machine] / 2
    if np.sum(sorted_distance > threshold * c_sd) == 0:
        truncation = distance_to_center.shape[0]
    else:
        truncation = np.where(sorted_distance > threshold * c_sd)[0][0]

    return indices[:truncation]


# -------------------------------------------------------------------
# split sample to different machines and fit mixture on each machine
# -------------------------------------------------------------------
def main(random_state, local_ss):

    n_split = 50
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
    letters_train_label = np.load(
        "preprocessed_data/NIST_label_letters_train.npy")

    print(digits_train.shape, digits_test.shape, letters_train.shape)

    shuffled_index = np.random.permutation(np.arange(digits_train.shape[0]))
    digits_train = digits_train[shuffled_index]

    for split in tqdm(range(n_split)):
        local = []
        for i in range(10):
            local.append(digits_train[digits_train_label == i][(
                split * local_per_class):((split + 1) * local_per_class)])
        local = np.vstack(local)
        # local = digits_train[(split * local_ss):((split + 1) * local_ss)]
        # print(local.shape)
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
        # gmmk = GaussianMixture(
        #     n_components=K,
        #     covariance_type="full",
        #     max_iter=10000,
        #     reg_covar=1.0 / np.sqrt(local.shape[0]),
        #     n_init=10,
        #     tol=1e-10,
        #     random_state=10,
        #     verbose=0,
        #     verbose_interval=1,
        #     init_params="k-means++",
        #     warm_start=True,
        # )

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
        # for failure_rate in [0.0]:
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

            # generate Byzantine failure index

            byzantine_machine_index = np.random.choice(n_split,
                                                       int(n_split *
                                                           failure_rate),
                                                       replace=False)
            print(byzantine_machine_index)
            for b_index in byzantine_machine_index:
                local = letters_train[np.random.choice(letters_train.shape[0],
                                                       local_ss,
                                                       replace=False)]
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
                # gmmk = GaussianMixture(
                #     n_components=K,
                #     covariance_type="full",
                #     max_iter=10000,
                #     reg_covar=1.0 / np.sqrt(local.shape[0]),
                #     n_init=10,
                #     tol=1e-10,
                #     random_state=10,
                #     verbose=0,
                #     verbose_interval=1,
                #     init_params="k-means++",
                #     warm_start=True,
                # )
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

            save_file = os.path.join(
                save_folder,
                "case_" + str(random_state) + "_nsplit_" + str(n_split) +
                "_ncomp_" + str(K) + "_d_" + str(D) + "_ss_" + str(local_ss) +
                "_failuretate_" + str(failure_rate) + ".pickle",
            )

            output_data = {
                "local_ARI": local_ARI,
                "local": (local_means, local_covs, local_weights),
                "local_ll": local_ll,
            }
            # for local_pmle_covs in local_covs:
            #     for i, cov in enumerate(local_pmle_covs):
            #         eigvals = np.linalg.eigvals(cov)
            #         print(i, eigvals.min(), eigvals.max())

            f = open(save_file, "wb")
            pickle.dump(output_data, f)
            f.close()

        else:
            byzantine_machine_index = []

        output_data = {}
        output_data["byzantine_machine"] = byzantine_machine_index

        # ------------------------------
        # COAT
        # ------------------------------
        start_time = time.time()
        which_GMM, pairwisedist = robustmedian(
            local_means,
            local_covs,
            local_weights,
            ground_distance="CTD-KL",
            # ground_distance="L2",
            coverage_ratio=0.5,
        )
        coat_time = time.time() - start_time

        # print(pairwisedist)

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
        # GMR using 50% around COAT
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
        # GMR by Thresholding (option1)
        # ------------------------------
        start_time = time.time()
        distance_to_center = pairwisedist[which_GMM]

        # ared_untruncated_indices = ared_threshold(distance_to_center,
        #                                           pairwisedist,
        #                                           local_ss * n_split,
        #                                           option=3)
        ared_untruncated_indices = ared_threshold(distance_to_center,
                                                  pairwisedist, which_GMM)
        # breakpoint()

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
        # GMR without trimming
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
        oracle_trimmed_weights = np.concatenate([
            local_weights[i] for i in range(n_split)
            if i not in byzantine_machine_index
        ])
        oracle_trimmed_weights /= int(n_split * (1 - failure_rate))

        oracle = GMR_CTD(
            np.concatenate([
                local_means[i] for i in range(n_split)
                if i not in byzantine_machine_index
            ]),
            np.concatenate([
                local_covs[i] for i in range(n_split)
                if i not in byzantine_machine_index
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
            "_failurerate_" + str(failure_rate) + ".pickle",
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

    args = parser.parse_args()
    local_ss = int(args.local_ss)
    seed = args.seed

    main(seed, local_ss)
