/**
 * Returns the dot product of two vectors.
 * @param {Array<Number>|Float64Array} a - Vector.
 * @param {Array<Number>|Float64Array} b - Vector.
 * @returns {Number} - The dot product of {@link a} and {@link b}.
 */
export default function (a, b) {
    let sum = 0;
    const N = a.length;
    for (let i = 0; i < N; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}