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
    let n = A.rows;
    const D = new Matrix(n, n);
    for (let i = 0; i < n; ++i) {
        const A_i = A.row(i);
        const D_i = D.row(i);
        for (let j = i + 1; j < n; ++j) {
            D.set_entry(j, i, D_i[j] = metric(A_i, A.row(j)));
        }
    }
    return D;
}
