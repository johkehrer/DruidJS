/**
 * Computes the chebyshev distance (L<sub>∞</sub>) between {@link a} and {@link b}.
 * @memberof module:metrics
 * @alias chebyshev
 * @param {Number[]} a
 * @param {Number[]} b
 * @returns {Number} the chebyshev distance between {@link a} and {@link b}.
 */
export default function (a, b) {
    let max = -1;
    const n = a.length;
    for (let i = 0; i < n; ++i) {
        const a_b = Math.abs(a[i] - b[i]);
        if (a_b > max) max = a_b;
    }
    return max;
}
