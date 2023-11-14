import { inner_product } from "../linear_algebra/index.js";
import { Matrix } from "../matrix/index.js";

/**
 * Computes the QR Decomposition of the Matrix `A` using Gram-Schmidt process.
 * @memberof module:linear_algebra
 * @alias qr
 * @param {Matrix} A
 * @returns {{R: Matrix, Q: Matrix}}
 * @see {@link https://en.wikipedia.org/wiki/QR_decomposition#Using_the_Gram%E2%80%93Schmidt_process}
 */
export default function (A) {
    const [rows, cols] = A.shape;
    const Q = new Matrix(rows, cols, "identity");
    const R = new Matrix(cols, cols, 0);
    const Q_val = Q.values;

    for (let j = 0; j < cols; ++j) {
        const v = A.col(j);
        for (let i = 0; i < j; ++i) {
            let q_dot_v = 0;
            for (let row = 0, k = i; row < rows; ++row, k += cols) {
                q_dot_v += Q_val[k] * v[row];
            }
            for (let row = 0, k = i; row < rows; ++row, k += cols) {
                v[row] -= Q_val[k] * q_dot_v;
            }
            R.set_entry(i, j, q_dot_v);
        }
        const v_norm = Math.sqrt(inner_product(v, v));
        for (let row = 0, k = j; row < rows; ++row, k += cols) {
            Q_val[k] = v[row] / v_norm;
        }
        R.set_entry(j, j, v_norm);
    }
    return { R, Q };
}
