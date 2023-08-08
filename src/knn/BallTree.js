import { euclidean } from "../metrics/index.js";
import { Heap } from "../datastructure/index.js";
import { quickselect } from "../util/index.js";

const LEAF_SIZE = 32;

class BTNode {
    constructor(pivot, radius = null, child1 = null, child2 = null) {
        this.pivot = pivot;
        this.child1 = child1;
        this.child2 = child2;
        this.radius = radius;
    }
}

class BTLeaf {
    constructor(pivot, radius, points) {
        this.pivot = pivot;
        this.radius = radius;
        this.points = points;
    }
}

/**
 * @class
 * @alias BallTree
 */
export class BallTree {
    /**
     * Generates a BallTree with given {@link elements}.
     * @constructor
     * @memberof module:knn
     * @alias BallTree
     * @param {Array=} elements - Elements which should be added to the BallTree
     * @param {Function} [metric = euclidean] metric to use: (a, b) => distance
     * @see {@link https://en.wikipedia.org/wiki/Ball_tree}
     * @see {@link https://github.com/invisal/noobjs/blob/master/src/tree/BallTree.js}
     * @returns {BallTree}
     */
    constructor(elements = null, metric = euclidean, add_distances = null) {
        this._addDist = add_distances || ((a, b) => a + b);
        this._metric = metric;
        if (elements) this.add(elements);
        return this;
    }

    /**
     * @param {Array<*>} elements - new elements.
     * @returns {BallTree}
     */
    add(elements) {
        const arr = elements.map((e, i) => ({ element: e, index: i }));
        this._root = this._construct(arr, 0, elements.length);
        return this;
    }

    /**
     * @private
     * @param {Array<*>} arr
     * @returns {BTNode} root of balltree.
     */
    _construct(arr, left, right) {
        // partially sort array according to dimension of greatest spread
        const p_idx = (left + right) >> 1;
        const c = this._greatest_spread(arr, left, right);
        quickselect(arr, (a, b) => a.element[c] - b.element[c], p_idx, left, right - 1);

        // compute radius of ball
        let radius = 0.0;
        const metric = this._metric;
        const pivot = arr[p_idx].element;
        for (let dist, i = left; i < right; ++i) {
            dist = metric(pivot, arr[i].element);
            if (dist > radius) radius = dist;
        }

        if (right - left > LEAF_SIZE) {
            const L = this._construct(arr, left, p_idx);
            const R = this._construct(arr, p_idx, right);
            return new BTNode(pivot, radius, L, R);
        }
        return new BTLeaf(pivot, radius, arr.slice(left, right));
    }

    /**
     * @private
     * @param {BTNode} arr
     * @returns {Number}
     */
    _greatest_spread(arr, left, right) {
        let maxDim = -1;
        let maxSpread = 0.0;
        const d = arr[left].element.length;
        for (let val, min, max, i = 0; i < d; ++i) {
            min = max = arr[left].element[i];
            for (let j = left + 1; j < right; ++j) {
                val = arr[j].element[i];
                if (val < min) min = val;
                if (val > max) max = val;
            }
            const spread = max - min;
            if (spread > maxSpread) {
                maxSpread = spread;
                maxDim = i;
            }
        }
        return maxDim;
    }

    /**
     * @param {*} t - query element.
     * @param {Number} [k = 5] - number of nearest neighbors to return.
     * @returns {Heap} - Heap consists of the {@link k} nearest neighbors.
     */
    search(t, k = 5) {
        const heap = new Heap(null, d => this._metric(d.element, t), "max");
        return this._search(t, k, heap, this._root);
    }

    /**
     * @private
     * @param {*} t - query element.
     * @param {Number} [k = 5] - number of nearest neighbors to return.
     * @param {Heap} Q - Heap consists of the currently found {@link k} nearest neighbors.
     * @param {BTNode|BTLeaf} B
     */
    _search(t, k, Q, B) {
        const metric = this._metric;
        if (Q.length >= k && metric(t, B.pivot) >= this._addDist(Q.first.value, B.radius)) {
            return Q;
        }
        if (B.points) {
            // B is leaf
            for (const p of B.points) {
                if (Q.length < k) Q.push(p);
                else Q.pushPop(p);
            }
        } else if (metric(t, B.child1.pivot) < metric(t, B.child2.pivot)) {
            // search the child node that is closest to t first
            this._search(t, k, Q, B.child1);
            this._search(t, k, Q, B.child2);
        } else {
            this._search(t, k, Q, B.child2);
            this._search(t, k, Q, B.child1);
        }
        return Q;
    }
}