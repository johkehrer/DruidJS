/**
 * Returns index and value of minimum in Array {@link values}.
 * @memberof module:utils
 * @alias min_index
 * @param {Array} values
 * @returns {Array}
 */
export default function (values) {
    let min_idx = 0;
    let min_val = values[0];
    const N = values.length;
    for (let val, i = 1; i < N; ++i) {
        val = values[i];
        if (min_val > val) {
            min_val = val, min_idx = i;
        }
    }
    return [min_idx, min_val];
}