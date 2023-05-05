/**
 * Returns index and value of maximum in Array {@link values}.
 * @memberof module:utils
 * @alias min_index
 * @param {Array} values
 * @returns {Array}
 */
export default function (values) {
    let max_idx = 0;
    let max_val = values[0];
    const N = values.length;
    for (let val, i = 1; i < N; ++i) {
        val = values[i];
        if (max_val < val) {
            max_val = val, max_idx = i;
        }
    }
    return [max_idx, max_val];
}