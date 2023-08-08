/**
 * Compute the sum of two squared euclidean distances (l<sub>2</sub><sup>2</sup>) <code>a</code> and <code>b</code>.
 * @memberof module:metrics
 * @alias euclidean_squared_sum
 * @param {Number} a
 * @param {Number} b
 * @returns {Number} the sum of two squared euclidean distances.
 */
export default function (a, b) {
    return a + b + 2 * Math.sqrt(a * b);
}