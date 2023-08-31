import { euclidean } from "../metrics/index.js";
import { Matrix } from "./index.js";

/**
 * Computes the distance matrix of datamatrix {@link A}.
 * @memberof module:matrix
 * @alias distance_matrix
 * @param {Matrix} A - Matrix.
 * @param {Function} [metric=euclidean] - The diistance metric.
 * @returns {Matrix} D - The distance matrix of {@link A}.
 */
export default function (A, metric = euclidean) {
    const N = A.rows;
    const D = new Matrix(N, N);
    const data = D.values;
    for (let i = 0; i < N; ++i) {
        const A_i = A.row(i);
        for (let j = i + 1, i_j = i * N + j, j_i = j * N + i; j < N; j_i += N) {
            data[j_i] = data[i_j++] = metric(A_i, A.row(j++));
        }
    }
    return D;
}
