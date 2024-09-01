import numpy as np
from dadapy import Data
import torch
from sklearn.metrics import adjusted_rand_score
import pickle
from collections import Counter
from collections import defaultdict
import scipy as sp
import matplotlib.pyplot as plt


def get_clusters(
    model_path,
    layer,
    mask,
    gtl,
    subjects,
    Z=1.6,
):

    X = torch.load(f"{model_path}/l{layer}_target.pt")
    X = X.to(torch.float64).numpy()[mask]

    # check we do not have opverlapping datapoints in case ramove tham from the relevant arrays
    X_, indx, inverse = np.unique(X, axis=0, return_inverse=True, return_index=True)
    sorted_indx = np.sort(indx)
    X_sub = X[sorted_indx]
    gtl = gtl[sorted_indx]
    subjects = subjects[sorted_indx]

    # mapping indices to subjects
    index_to_subject = {}
    for i in range(len(gtl)):
        if gtl[i] in index_to_subject:
            assert index_to_subject[gtl[i]] == subjects[i], (i, gtl[i], subjects[i])
        else:
            index_to_subject[gtl[i]] = subjects[i]

    # ************************************************************************
    # here we should try also halo = True

    d = Data(coordinates=X_sub, maxk=300)
    ids, _, _ = d.return_id_scaling_gride(range_max=300)
    print(ids)
    d.set_id(ids[3])
    d.compute_density_kNN(k=16)
    cluster_assignment = d.compute_clustering_ADP(Z=Z, halo=False)
    is_core = cluster_assignment != -1

    print("n_clusters", d.N_clusters)
    print(adjusted_rand_score(gtl[is_core], cluster_assignment[is_core]))
    return d, index_to_subject


def get_dissimilarity(d, min_population=30, threshold=99999):

    # **************************************************************
    # we consider only clusters with at least 30 points (this can be relaxed or decreased)
    min_population = 30

    cluster_mask = []
    final_clusters_tmp = []
    log_den_tmp = []
    for c, cluster_indices in zip(d.cluster_centers, d.cluster_indices):
        if len(cluster_indices) > min_population and d.log_den[c] < threshold:
            cluster_mask.append(True)
            final_clusters_tmp.append(cluster_indices)
            log_den_tmp.append(d.log_den[c])
        else:
            cluster_mask.append(False)

    cluster_mask = np.array(cluster_mask)
    assert cluster_mask.shape[0] == d.N_clusters

    final_clusters = final_clusters_tmp
    nclus = len(final_clusters)
    assert nclus == np.sum(cluster_mask)
    saddle_densities = d.log_den_bord[cluster_mask]
    saddle_densities = saddle_densities[:, cluster_mask]
    density_peak_indices = np.array(d.cluster_centers)[cluster_mask]

    # *****************************************************
    # pivotal similrity matrices of the clusters with a population > min_population
    density_peaks = d.log_den[density_peak_indices]

    Fmax = np.max(density_peaks)
    peak_y = density_peaks - Fmax

    Dis = []
    for i in range(nclus):
        for j in range(i + 1, nclus):
            Dis.append(Fmax - saddle_densities[i][j])

    # similar clusters are those whic are closer "small distance".
    # we subtract the saddle point density to the max density peak highest saddles --> close categories
    Dis = np.array(Dis)
    return Dis, peak_y, final_clusters


def get_xy_coords(
    dis,
    final_clusters,
    index_to_subject,
    gtl,
    peak_y,
    max_displayed_names=2,
):
    DD = sp.cluster.hierarchy.linkage(dis, method="weighted")
    # ************************************************************************
    # get subjects in the final clusters

    # we consider a subject relevant in a cluster only if it has a frequancy 0.2*the most present class
    subject_relevance = 0.2
    clust_subjects = defaultdict(list)
    clust_approx_subjects = defaultdict(list)
    clust_percent = defaultdict(list)

    # **********************************************************************************************

    for i in range(len(final_clusters)):
        # sub, comp, approx = get_composition_imbalanced(
        #     final_clusters[i], index_to_subject, subject_relevance
        # )

        sub, comp, approx = get_composition_imbalanced(
            gtl, final_clusters[i], index_to_subject, subject_relevance=0.2
        )

        clust_subjects[i] = sub
        clust_approx_subjects[i] = approx
        clust_percent[i] = comp

    labels = []
    for i, (clust, subjects) in enumerate(clust_approx_subjects.items()):
        name = ""
        for i, sub in enumerate(subjects):
            if i < max_displayed_names:
                name += f" {sub},"
        labels.append(name.strip()[:-1])

    # without labels
    dn = sp.cluster.hierarchy.dendrogram(
        DD,
        p=35,
        truncate_mode=None,
        color_threshold=24,
        get_leaves=True,
        orientation="bottom",
        labels=None,
        above_threshold_color="b",
    )

    xcoords = np.array(dn["icoord"]) / 10 - 0.5

    # integer label of the leaf nodes (representing the clusters)
    order = np.array([int(dn["ivl"][i]) for i in range(len(dn["ivl"]))])
    # density of the peaks
    reordered_peaks = peak_y[order]

    xcoords = np.array(dn["icoord"]) / 10 - 0.5
    ycoords = -np.array(dn["dcoord"])

    # reassign ycoords
    ycoords[np.where(ycoords[:, 0] == 0), 0] = reordered_peaks[
        (xcoords[np.where(ycoords[:, 0] == 0), 0][0]).astype(int)
    ]
    ycoords[np.where(ycoords[:, 3] == 0), 3] = reordered_peaks[
        (xcoords[np.where(ycoords[:, 3] == 0), 3][0]).astype(int)
    ]
    return xcoords, ycoords


###############################################################
##############################################################à


def get_composition_imbalanced(
    gtl, final_clusters, index_to_subject, subject_relevance=0.2
):
    cluster_subject_samples = Counter(gtl[np.array(final_clusters)])

    total_subject_samples = Counter(gtl)

    freq_in_cluster = {
        key: val / total_subject_samples[key]
        for key, val in cluster_subject_samples.items()
    }

    subject_fraction = {
        k: v
        for k, v in sorted(
            freq_in_cluster.items(), key=lambda item: item[1], reverse=True
        )
    }
    population_fraction = {
        # k: v / len(cluster_indices)
        k: v / len(final_clusters)
        for k, v in sorted(
            cluster_subject_samples.items(), key=lambda item: item[1], reverse=True
        )
    }
    clust_subjects = [index_to_subject[key] for key in subject_fraction.keys()]

    class_to_count = []
    highest_fraction = max(list(subject_fraction.values()))
    # dumb_check
    for key, val in subject_fraction.items():
        assert highest_fraction == val
        break

    for i, (index, fraction) in enumerate(subject_fraction.items()):
        # current_perc += clust_percent[i]
        # we consider a subject relevant only iif its presence is >0.2 the maximum popolated class
        if (
            fraction / highest_fraction > subject_relevance
            or population_fraction[index] > 0.5
        ):
            class_to_count.append(clust_subjects[i])

    if len(class_to_count) > 4:
        class_to_count = [f"mix of {len(class_to_count)} subjects"]

    return clust_subjects, subject_fraction, class_to_count


def get_composition(cluster_indices, index_to_subject, subject_relevance):
    # subject integer identifier: frequency of the subjects
    counts = Counter(np.array(cluster_indices) // 100)
    # total_number of points in the cluster
    population = len(cluster_indices)

    # all subjects contained in a cluster
    clust_subjects = [index_to_subject[t[0]] for t in counts.most_common()]

    clust_percent = [t[1] / population for t in counts.most_common()]

    assert np.sum(clust_percent) > 0.9999, (clust_percent, np.sum(clust_percent))

    class_to_count = []
    current_perc = 0

    #
    for i in range(len(clust_subjects)):
        current_perc += clust_percent[i]
        # we consider a subject relevant only iif its presence is >0.2 the maximum popolated class
        if clust_percent[i] / clust_percent[0] > subject_relevance:
            class_to_count.append(clust_subjects[i])
        if current_perc > 0.9:
            break

    if clust_percent[0] < 0.3 or len(class_to_count) > 3:
        class_to_count = [f"mix of {len(class_to_count)} subjects"]

    return clust_subjects, clust_percent, class_to_count


def get_subject_array(base_dir):
    subject_path = f"{base_dir}/results/mmlu/llama-3-8b/5shot"
    with open(f"{subject_path}/statistics_target.pkl", "rb") as f:
        stats = pickle.load(f)
    return np.array(stats["subjects"])


def get_dataset_mask(base_dir, subjects, nsample_subjects=100):

    if nsample_subjects == 100:
        mask_path = f"{base_dir}/diego/analysis"
        mask = np.load(f"{mask_path}/test_mask_100.npy")
        gtl = np.repeat(np.arange(57), nsample_subjects)

        # double check
        frequences = Counter(subjects[mask]).values()
        assert len(np.unique(list(frequences))) == 1
        assert np.unique(list(frequences))[0] == nsample_subjects, np.unique(
            list(frequences)
        )[0]
    if nsample_subjects == 200:
        mask_path = f"{base_dir}/diego/analysis"
        mask = np.load(f"{mask_path}/test_mask_200.npy")
        subjects = subjects[mask]
        sub_to_int = {sub: i for i, sub in enumerate(np.unique(subjects))}
        gtl = [sub_to_int[sub] for sub in subjects]

    else:

        mask = []
        gtl = []
        for i, sub in enumerate(np.unique(subjects)):
            ind = np.where(sub == subjects)[0]
            nsamples = min(nsample_subjects, len(ind))
            chosen = np.random.choice(ind, nsamples, replace=False)
            if sub == "world_religions":
                chosen = np.where(subjects == "world_religions")[0][
                    : (min(nsamples, 169))
                ]
            mask.extend(list(np.sort(chosen)))
            gtl.extend([i for _ in range(len(chosen))])

    return np.array(mask), np.array(gtl)


############################################à


def plot_with_labels(
    dis,
    final_clusters,
    ground_truth_labels,
    index_to_subject,
    ori="left",
    path=None,
    width=9,
    color_threshold=10,
    height=14,
    leaf_font_size=10,
):
    # ****************************************************************************************************************
    DD = sp.cluster.hierarchy.linkage(dis, method="weighted")
    # we consider a subject relevant in a cluster only if it has a frequancy 0.2*the most present class
    clust_subjects = defaultdict(list)
    clust_approx_subjects = defaultdict(list)
    clust_percent = defaultdict(list)

    for i in range(len(final_clusters)):
        sub, comp, approx = get_composition_imbalanced(
            ground_truth_labels,
            final_clusters[i],
            index_to_subject,
            subject_relevance=0.2,
        )
        clust_subjects[i] = sub
        clust_approx_subjects[i] = approx
        clust_percent[i] = comp
    labels = []
    for i, (clust, subjects) in enumerate(clust_approx_subjects.items()):
        name = ""
        for sub in subjects:
            name += f" {sub},"
        labels.append(name.strip()[:-1])

    # ************************************
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height))
    dn = sp.cluster.hierarchy.dendrogram(
        DD,
        p=30,
        truncate_mode=None,
        color_threshold=color_threshold,
        get_leaves=True,
        orientation=ori,
        above_threshold_color="C5",
        labels=labels,
        leaf_font_size=leaf_font_size,
    )
    ax.set_xticks([])
    ax.set_xticklabels([])

    plt.tight_layout()
    if path is not None:
        plt.savefig(f"{path}.png")
        plt.savefig(f"{path}.pdf")
