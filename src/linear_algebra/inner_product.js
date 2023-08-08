/**
 * Computes the inner product between two arrays of the same length.
 * @memberof module:linear_algebra
 * @alias inner_product
 * @param {Array|Float64Array} a - Array a
 * @param {Array|Float64Array} b - Array b
 * @returns The inner product between {@link a} and {@link b}
 */
export default function (a, b) {
    let sum = 0;
    const N = a.length;
    for (let i = 0; i < N; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
