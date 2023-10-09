import { distance_matrix } from "../matrix/index.js";
import { quickselect } from "../util/index.js";
import { euclidean } from "../metrics/index.js";

/**
 * Computes the k-nearest neighbors of each row of {@link A}.
 * @memberof module:matrix
 * @alias k_nearest_neigbhors
 * @param {Matrix} A - Either the data matrix, or a distance matrix.
 * @param {Number} k - The number of neighbors to compute.
 * @param {Function|"precomputed"} [metric=euclidean]
 * @returns {Array<Object>} -
 */
export default function (A, k, metric = euclidean) {
    const N = A.rows;
    const D = metric == "precomputed" ? A : distance_matrix(A, metric);
    const nN = Array.from({ length: N }, (_, row) => {
        const arr = Array.from(D.row(row), (d, col) => ({ i: row, j: col, distance: d }));
        arr[row] = arr[0]; // elliminate diagonal
        if (k < N) quickselect(arr, (a, b) => a.distance - b.distance, k, 1, N - 1);
        return arr.slice(1, k + 1);
    });
    return nN;
}


