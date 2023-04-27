/**
 * Creates an Array containing {@link number} numbers from {@link start} to {@link end}.
 * If <code>{@link number} = null</null>.
 * @memberof module:matrix
 * @alias linspace
 * @param {Number} start - Start value.
 * @param {Number} end - End value.
 * @param {Number} [number = null] - Number of number between {@link start} and {@link end}.
 * @returns {Array} - An array with {@link number} entries, beginning at {@link start} ending at {@link end}.
 */
export default function (start, end, number = null) {
    if (!number) {
        number = Math.max(Math.round(end - start) + 1, 1);
    }
    if (number < 2) {
        return number === 1 ? [start] : [];
    }
    const step = (end - start) / (number - 1);
    const result = Array.from({ length: number }, (_, i) => start + i * step);
    return result;
}
