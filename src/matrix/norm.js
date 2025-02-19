import { inner_product } from "../linear_algebra/index.js";
import { euclidean } from "../metrics/index.js";
import { Matrix } from "../matrix/index.js";

/**
 * Computes the norm of a vector, by computing its distance to **0**.
 * @memberof module:matrix
 * @alias norm
 * @param {Matrix|Array<Number>|Float64Array} v - Vector.
 * @param {Function} [metric = euclidean] - Which metric should be used to compute the norm.
 * @returns {Number} - The norm of {@link v}.
 */
export default function (v, metric = euclidean) {
    let vector = null;
    if (v instanceof Matrix) {
        if (v.rows === 1 || v.cols === 1) vector = v.values;
        else throw new Error("Matrix must be 1d!");
    } else {
        vector = v;
    }
    if (metric === euclidean) {
        return Math.sqrt(inner_product(vector, vector));
    }
    const n = vector.length;
    const zeros = new Float64Array(n);
    return metric(vector, zeros);
}
