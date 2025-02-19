/**
 * @class
 * @alias Heap
 */
export class Heap {
    /**
     * A heap is a datastructure holding its elements in a specific way, so that the top element would be the first entry of an ordered list.
     * @constructor
     * @memberof module:datastructure
     * @alias Heap
     * @param {Array=} elements - Contains the elements for the Heap. {@link elements} can be null.
     * @param {Function} [accessor = (d) => d] - Function returns the value of the element.
     * @param {("min"|"max"|Function)} [comparator = "min"] - Function returning true or false defining the wished order of the Heap, or String for predefined function. ("min" for a Min-Heap, "max" for a Max_heap)
     * @returns {Heap}
     * @see {@link https://en.wikipedia.org/wiki/Binary_heap}
     */
    constructor(elements = null, accessor = d => d, comparator = "min") {
        if (elements) {
            return Heap.heapify(elements, accessor, comparator);
        } else {
            this._accessor = accessor;
            this._container = [];
            if (comparator == "min") {
                this._comparator = (a, b) => a < b;
            } else if (comparator == "max") {
                this._comparator = (a, b) => a > b;
            } else {
                this._comparator = comparator;
            }
            return this
        }
    }

    /**
     * Creates a Heap from an Array
     * @param {Array|Set} elements - Contains the elements for the Heap.
     * @param {Function=} [accessor = (d) => d] - Function returns the value of the element.
     * @param {(String=|Function)} [comparator = "min"] - Function returning true or false defining the wished order of the Heap, or String for predefined function. ("min" for a Min-Heap, "max" for a Max_heap)
     * @returns {Heap}
     */
    static heapify(elements, accessor = d => d, comparator = "min") {
        const heap = new Heap(null, accessor, comparator);
        const container = heap._container;
        for (const e of elements) {
            container.push({
                "element": e,
                "value": accessor(e),
            });
        }
        for (let i = Math.floor((elements.length / 2) - 1); i >= 0; --i) {
            heap._heapify_down(i);
        }
        return heap;
    }

    /**
     * Swaps elements of container array.
     * @private
     * @param {Number} index_a 
     * @param {Number} index_b 
     */
    _swap(index_a, index_b) {
        const container = this._container;
        [container[index_b], container[index_a]] = [container[index_a], container[index_b]];
        return;
    }

    /**
     * @private
     */
    _heapify_up() {
        const comparator = this._comparator;
        const container = this._container;
        let index = container.length - 1;
        let parent = (index - 1) >> 1;
        while (index > 0 && comparator(container[index].value, container[parent].value)) {
            this._swap(parent, index)
            index = parent;
            parent = (index - 1) >> 1;
        }
    }

    /**
     * Pushes the element to the heap.
     * @param {} element
     * @returns {Heap}
     */
    push(element, value = null) {
        const node = { element, value: value ?? this._accessor(element)};
        this._container.push(node);
        this._heapify_up();
        return this;
    }

    /**
     * Pushes a new element to the heap. At the same time it removes and returns the
     * top entry of the heap. This is faster than calling push() and pop() separately.
     * @param {} element
     * @returns {Object} Object consists of the element and its value (computed by {@link accessor}).
     */
    pushPop(element, value = null) {
        value = value ?? this._accessor(element);
        if (this._comparator(this._container[0].value, value)) {
            const item = this._container[0];
            this._container[0] = { element, value };
            this._heapify_down();
            return item;
        }
        return { element, value };
    }

    /**
     * @private
     * @param {Number} [index = 0]
     */
    _heapify_down(index = 0) {
        const container = this._container;
        const comparator = this._comparator;
        const length = container.length;
        let swapIdx = index;
        while (true) {
            const left = (index << 1) + 1;
            const right = left + 1;
            if (left < length && comparator(container[left].value, container[swapIdx].value)) {
                swapIdx = left;
            }
            if (right < length && comparator(container[right].value, container[swapIdx].value)) {
                swapIdx = right;
            }
            if (swapIdx === index) break;
            this._swap(index, swapIdx);
            index = swapIdx;
        }
    }

    /**
     * Removes and returns the top entry of the heap.
     * @returns {Object} Object consists of the element and its value (computed by {@link accessor}).
     */
    pop() {
        const container = this._container;
        if (container.length === 0) {
            return null;
        } else if (container.length === 1) {
            return container.pop();
        }
        this._swap(0, container.length - 1);
        const item = container.pop();
        this._heapify_down();
        return item;
    }

    /**
     * Returns the top entry of the heap without removing it.
     * @returns {Object} Object consists of the element and its value (computed by {@link accessor}).
     */
    get first() {
        return this._container.length > 0 ? this._container[0] : null;
    }


    /**
     * Yields the raw data
     * @yields {Object} Object consists of the element and its value (computed by {@link accessor}).
     */
    * iterate() {
        for (let i = 0, n = this._container.length; i < n; ++i) {
            yield this._container[i].element;
        }
    }

    /**
     * Returns the heap as ordered array.
     * @returns {Array} Array consisting the elements ordered by {@link comparator}.
     */
    toArray() {
        return this.data()
            .sort((a,b) => this._comparator(a, b) ? -1 : 0)
    }

    /**
     * Returns elements of container array.
     * @returns {Array} Array consisting the elements.
     */
    data() {
        return this._container
            .map(d => d.element)
    }

    /**
     * Returns the container array.
     * @returns {Array} The container array.
     */
    raw_data() {
        return this._container;
    }

    /**
     * The size of the heap.
     * @returns {Number}
     */
    get length() {
        return this._container.length;
    }

    /**
     * Returns false if the the heap has entries, true if the heap has no entries.
     * @returns {Boolean}
     */
    get empty() {
        return this.length === 0;
    }
}