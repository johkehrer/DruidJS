import { Matrix } from "../matrix/index.js";

/**
 * @class
 * @memberof module:utils
 * @alias Randomizer
 */
export class Randomizer {
    /**
     * Mersenne Twister random number generator.
     * @constructor
     * @param {Number} [_seed=new Date().getTime()] - The seed for the random number generator. If <code>_seed == null</code> then the actual time gets used as seed.
     * @see https://github.com/bmurray7/mersenne-twister-examples/blob/master/javascript-mersenne-twister.js
     */
    constructor(_seed) {
        this._N = 624;
        this._M = 397;
        this._MATRIX_A = 0x9908b0df;
        this._UPPER_MASK = 0x80000000;
        this._LOWER_MASK = 0x7fffffff;
        this._mt = new Array(this._N);
        this._mti = this.N + 1;

        this.seed = _seed || new Date().getTime();
        return this;
    }

    set seed(_seed) {
        this._seed = _seed;
        const mt = this._mt;

        mt[0] = _seed >>> 0;
        for (let mti = 1; mti < this._N; ++mti) {
            let s = mt[mti - 1] ^ (mt[mti - 1] >>> 30);
            mt[mti] = ((((s & 0xffff0000) >>> 16) * 1812433253) << 16) + (s & 0x0000ffff) * 1812433253 + mti;
            mt[mti] >>>= 0;
        }
        this._mti = this._N;
    }

    /**
     * Returns the seed of the random number generator.
     * @returns {Number} - The seed.
     */
    get seed() {
        return this._seed;
    }

    /**
     * Returns a float between 0 and 1.
     * @returns {Number} - A random number between [0, 1]
     */
    get random() {
        return this.random_int * (1.0 / 4294967296.0);
    }

    /**
     * Returns an integer between 0 and MAX_INTEGER.
     * @returns {Integer} - A random integer.
     */
    get random_int() {
        let y,
            mag01 = new Array(0x0, this._MATRIX_A);
        if (this._mti >= this._N) {
            let kk;

            /* if (this._mti == this._N + 1) {
                this.seed = 5489;
            } */

            const N_M = this._N - this._M;
            const M_N = this._M - this._N;
            const mt = this._mt;

            for (kk = 0; kk < N_M; ++kk) {
                y = (mt[kk] & this._UPPER_MASK) | (mt[kk + 1] & this._LOWER_MASK);
                mt[kk] = mt[kk + this._M] ^ (y >>> 1) ^ mag01[y & 0x1];
            }
            for (; kk < this._N - 1; ++kk) {
                y = (mt[kk] & this._UPPER_MASK) | (mt[kk + 1] & this._LOWER_MASK);
                mt[kk] = mt[kk + M_N] ^ (y >>> 1) ^ mag01[y & 0x1];
            }

            y = (mt[this._N - 1] & this._UPPER_MASK) | (mt[0] & this._LOWER_MASK);
            mt[this._N - 1] = mt[this._M - 1] ^ (y >>> 1) ^ mag01[y & 0x1];

            this._mti = 0;
        }

        y = this._mt[(this._mti += 1)];
        y ^= y >>> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >>> 18;

        return y >>> 0;
    }

    gauss_random() {
        let x, y, r;
        if (this._val != null) {
            x = this._val, this._val = null;
            return x;
        } else do {
            x = 2 * this.random - 1;
            y = 2 * this.random - 1;
            r = x * x + y * y;
        } while (!r || r > 1);
        const c = Math.sqrt(-2 * Math.log(r) / r);
        this._val = y * c; // cache this for next function call for efficiency
        return x * c;
    }

    /**
     * Returns samples from an input Matrix or Array.
     * @param {Matrix|Array|Float64Array} A - The input Matrix or Array.
     * @param {Number} n - The number of samples.
     * @returns {Array} - A random selection form {@link A} of {@link n} samples.
     */
    choice(A, n) {
        if (A instanceof Matrix) {
            const sample = this._choice(n, A.rows);
            return sample.map((d) => A.row(d));
        } else if (Matrix.isArray(A)) {
            const sample = this._choice(n, A.length);
            return sample.map((d) => A[d]);
        } else {
            throw new Error('A must be array or matrix');
        }
    }

    _choice(n, len) {
        if (n > len) {
            throw new Error("n bigger than A!");
        }
        const indices = Array.from({ length: len }, (_, i) => i);
        const samples = Array.from({ length: n }, () => {
            const random_index = this.random_int % len--;
            return indices.splice(random_index, 1)[0];
        });
        return samples;
    }

    /**
     * @static
     * Returns samples from an input Matrix or Array.
     * @param {Matrix|Array|Float64Array} A - The input Matrix or Array.
     * @param {Number} n - The number of samples.
     * @param {Number} seed - The seed for the random number generator.
     * @returns {Array} - A random selection form {@link A} of {@link n} samples.
     */
    static choice(A, n, seed = 1212) {
        const R = new Randomizer(seed);
        return R.choice(A, n);
        /* let rows = A.shape[0];
        if (n > rows) {
            throw new Error("n bigger than A!");
        }
        let rand = new Randomizer(seed);
        let sample = new Array(n);
        let index_list = linspace(0, rows - 1);
        for (let i = 0, l = index_list.length; i < n; ++i, --l) {
            let random_index = rand.random_int % l;
            sample[i] = index_list.splice(random_index, 1)[0];
        }
        //return result;
        //return new Matrix(n, cols, (row, col) => A.entry(sample[row], col))
        return sample.map((d) => A.row(d)); */
    }
}
