// Based on https://github.com/mourner/quickselect
// ISC license, Copyright 2018 Vladimir Agafonkin.
export default function quickselect(arr, compare, k, left = 0, right = arr.length - 1) {
  while (right > left) {
    if (right - left > 600) {
      const n = right - left + 1;
      const m = k - left + 1;
      const z = Math.log(n);
      const s = 0.5 * Math.exp(2 * z / 3);
      const sd = 0.5 * Math.sqrt(z * s * (n - s) / n) * (m - 0.5 * n < 0 ? -1 : 1);
      const newLeft = Math.max(left, Math.floor(k - m * s / n + sd));
      const newRight = Math.min(right, Math.floor(k + (n - m) * s / n + sd));
      quickselect(arr, compare, k, newLeft, newRight);
    }

    const t = arr[k];
    let i = left;
    let j = right;

    swap(arr, left, k);
    if (compare(arr[right], t) > 0) swap(arr, left, right);

    while (i < j) {
      swap(arr, i++, j--);
      while (compare(arr[i], t) < 0) ++i;
      while (compare(arr[j], t) > 0) --j;
    }

    if (compare(arr[left], t) === 0) swap(arr, left, j);
    else swap(arr, ++j, right);

    if (j <= k) left = j + 1;
    if (k <= j) right = j - 1;
  }

  return arr;
}

function swap(arr, i, j) {
  const t = arr[i];
  arr[i] = arr[j];
  arr[j] = t;
}