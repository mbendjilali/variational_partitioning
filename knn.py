import torch
from FRNN.frnn import frnn


def knn_1(
    xyz: torch.Tensor,
    k,
    r_max=1,
    oversample=False,
    self_is_neighbor=False,
):
    """Search k-NN inside for a 3D point cloud xyz. This search differs
    from `knn_2` in that it operates on a single cloud input (search and
    query are the same) and it allows oversampling the neighbors when
    less than `k` neighbors are found within `r_max`
    """
    assert k >= 1
    assert xyz.dim() == 2

    # Data initialization
    device = xyz.device
    xyz_query = xyz.view(1, -1, 3)
    xyz_search = xyz.view(1, -1, 3)
    if not xyz.is_cuda:
        xyz_query = xyz_query.cuda()
        xyz_search = xyz_search.cuda()

    # KNN on GPU. Actual neighbor search now
    k_search = k if self_is_neighbor else k + 1
    distances, neighbors, _, _ = frnn.frnn_grid_points(
        xyz_query,
        xyz_search,
        K=k_search,
        r=r_max,
    )

    # Remove each point from its own neighborhood
    neighbors = neighbors[0] if self_is_neighbor else neighbors[0][:, 1:]
    distances = distances[0] if self_is_neighbor else distances[0][:, 1:]

    # Oversample the neighborhoods where less than k points were found
    if oversample:
        neighbors, distances = oversample_partial_neighborhoods(
            neighbors,
            distances,
            k,
        )

    # Restore the neighbors and distances to the input device
    if neighbors.device != device:
        neighbors = neighbors.to(device)
        distances = distances.to(device)

    # Warn the user of partial and empty neighborhoods
    num_nodes = neighbors.shape[0]
    n_missing = (neighbors < 0).sum(dim=1)
    n_partial = (n_missing > 0).sum()
    n_empty = (n_missing == k).sum()
    if n_partial == 0:
        return neighbors, distances

    # print(
    #     f"\nWarning: {n_partial}/{num_nodes} points have partial "
    #     f"neighborhoods and {n_empty}/{num_nodes} have empty "
    #     f"neighborhoods (missing neighbors are indicated by -1 indices)."
    # )

    return neighbors, distances


def oversample_partial_neighborhoods(neighbors, distances, k):
    """Oversample partial neighborhoods with less than k points. Missing
    neighbors are indicated by the "-1" index.

    Remarks
      - Neighbors and distances are assumed to be sorted in order of
      increasing distance
      - All neighbors are assumed to have at least one valid neighbor.
      See `search_outliers` to remove points with not enough neighbors
    """
    # Initialization
    assert neighbors.dim() == distances.dim() == 2
    device = neighbors.device

    # Get the number of found neighbors for each point. Indeed,
    # depending on the cloud properties and the chosen K and radius,
    # some points may receive `-1` neighbors
    n_found_nn = (neighbors != -1).sum(dim=1)

    # Identify points which have more than k_min and less than k
    # neighbors within R. For those, we oversample the neighbors to
    # reach k
    idx_partial = torch.where(n_found_nn < k)[0]
    nbors_partial = neighbors[idx_partial]
    dist_partial = distances[idx_partial]

    # Since the neighbors are sorted by increasing distance, the missing
    # neighbors will always be the last ones. This helps finding their
    # number and position, for oversampling.

    # *******************************************************************
    # The above statement is actually INCORRECT because the outlier
    # removal may produce "-1" neighbors at unexpected positions. So
    # either we manage to treat this in a clean vectorized way, or we
    # fall back to the 2-searches solution...
    # Honestly, this feels like it is getting out of hand, let's keep
    # things simple, since we are not going to save so much computation
    # time with KNN wrt the partition.
    # *******************************************************************

    # For each missing neighbor, compute the size of the discrete set to
    # oversample from.
    n_valid = n_found_nn[idx_partial]
    n_valid = n_valid.repeat_interleave(k - n_found_nn[idx_partial])

    # Compute the oversampling row indices.
    idx_x_sampling = torch.arange(
        nbors_partial.shape[0],
        device=device,
    ).repeat_interleave(k - n_found_nn[idx_partial])

    # Compute the oversampling column indices. The 0.9999 factor is a
    # security to handle the case where torch.rand is to close to 1.0,
    # which would yield incorrect sampling coordinates that would in
    # result in sampling '-1' indices (ie all we try to avoid here)
    idx_y_sampling = n_valid * torch.rand(n_valid.shape[0], device=device)
    idx_y_sampling *= 0.9999
    idx_y_sampling.floor().long()

    # Apply the oversampling
    idx_missing = torch.where(nbors_partial == -1)
    nbors_partial[idx_missing] = nbors_partial[idx_x_sampling, idx_y_sampling]
    dist_partial[idx_missing] = dist_partial[idx_x_sampling, idx_y_sampling]

    # Restore the oversampled neighborhoods with the rest
    neighbors[idx_partial] = nbors_partial
    distances[idx_partial] = dist_partial

    return neighbors, distances
