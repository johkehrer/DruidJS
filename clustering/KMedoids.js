import { euclidean } from "../metrics/index.js";
import { distance_matrix, Matrix } from "../matrix/index.js";
import { min, min_index, quickselect, Randomizer } from "../util/index.js";
/**
 * @class
 * @alias KMedoids
 */
export class KMedoids {
    /**
     * @constructor
     * @memberof module:clustering
     * @alias KMedoids
     * @todo needs restructuring. 
     * @param {Matrix} matrix - data matrix
     * @param {Numbers} K - number of clusters
     * @param {number} [max_iter=null] - maximum number of iterations. Default is 10 * Math.log10(N)
     * @param {Function} [metric = euclidean] - metric defining the dissimilarity 
     * @param {Number} [seed = 1212] - seed value for random number generator
     * @returns {KMedoids}
     * @see {@link https://link.springer.com/chapter/10.1007/978-3-030-32047-8_16} Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms
     */
    constructor(matrix, K, max_iter=null, metric = euclidean, seed=1212) {
        this._metric = metric;
        this._matrix = matrix;
        const [N, D] = matrix.shape;
        if (K > N) K = N;
        this._K = K;
        this._N = N;
        this._D = D;
        this._max_iter = max_iter || 10 * Math.log10(N);
        this._distance_matrix = distance_matrix(matrix, metric);
        this._randomizer = new Randomizer(seed);
        this._cluster_medoids = this._get_random_medoids(K);
        //if (init) this.init(K, this._cluster_medoids);
        this._is_initialized = false;
        return this;
    }

    /**
     * @returns {Array<Array>} - Array of clusters with the indices of the rows in given {@link matrix}.
     */
    get_clusters() {
        const N = this._N;
        const K = this._K;
        if (!this._is_initialized) {
            this.init(K, this._cluster_medoids);
        }
        const result = Array.from({ length: K }, () => []);
        for (let j = 0; j < N; ++j) {
            result[this._nearest_medoid(j).index_nearest].push(j);
        }
        result.medoids = this._cluster_medoids;
        return result;
    }

    async* generator() {
        const max_iter = this._max_iter;
        yield this.get_clusters();
        let finish = false;
        let i = 0;
        do {
            finish = this._iteration();
            yield this.get_clusters();
        } while (!finish && ++i < max_iter)
    }

    /**
     * Algorithm 1. FastPAM1: Improved SWAP algorithm
     */
    /* _iteration_1() {
        const A = this._A;
        const N = this._N;
        const K = this._K;
        const medoids = this._cluster_medoids;
        let DeltaTD = 0;
        let m0 = null;
        let x0 = null;
        A.forEach((x_j, j) => {
            if (medoids.findIndex(m => m === j) < 0) {
                const nearest_medoid = this._nearest_medoid(x_j, j);
                const d_j = nearest_medoid.distance_nearest; // distance to current medoid
                const deltaTD = new Array(K).fill(-d_j); // change if making j a medoid
                A.forEach((x_o, o) => {
                    // disance to new medoid
                    const d_oj = this._get_distance(o, j, x_o, x_j);
                    const {
                        "index_nearest": n,
                        "distance_nearest": d_n,
                        "distance_second": d_s,
                    } = this._nearest_medoid(x_o, o);
                    this._clusters[o] = n; // cached values
                    deltaTD[n] += Math.min(d_oj, d_s) - d_n; // loss change
                    if (d_oj < d_n) { // reassignment check
                        deltaTD.forEach((d_i, i) => {
                            if (n !== i) {
                                deltaTD[i] = d_i + d_oj - d_n; // update loss change
                            }
                        });
                    }
                });
                // choose best medoid i;
                const i = deltaTD
                    .map((d, i) => [d, i])
                    .sort((d1, d2) => d1[0] - d2[0])[0][1];
                const deltaTD_i = deltaTD[i];
                // store
                if (deltaTD_i < DeltaTD) {
                    DeltaTD = deltaTD_i;
                    m0 = i;
                    x0 = j;
                }
            }
        });

        if (DeltaTD >= 0) {
            return true // break loop if DeltaTD >= 0
        }
        // swap roles of medoid m and non-medoid x;
        medoids[m0] = x0;
        this._cluster_medoids = medoids;
        return false
    } */

    /** Algorithm 2. FastPAM2: SWAP with multiple candidates
     *
     */
    _iteration() {
        const N = this._N;
        const K = this._K;
        const D = this._distance_matrix;
        const medoids = this._cluster_medoids;
        const cache = Array.from({ length: N }, (_, i) => this._nearest_medoid(i));
        // empty best candidates array
        const DeltaTD = new Float64Array(K);
        const xs = Array.from({ length: K });
        for (let j = 0; j < N; ++j) {
            if (medoids.indexOf(j) < 0) { // x_j not in medoids
                const d_j = cache[j].distance_nearest; // distance to current medoid
                const deltaTD = new Float64Array(K).fill(-d_j); // change if making j a medoid
                for (let o = 0; o < N; ++o) {
                    if (j === o) continue;
                    const d_oj = D.entry(o, j); // distance to new medoid
                    const {"index_nearest": n, "distance_nearest": d_n, "distance_second": d_s} = cache[o]; // cached
                    deltaTD[n] += Math.min(d_oj, d_s) - d_n; // loss change for x_o
                    // Reassignment check
                    if (d_oj < d_n) {
                        // update loss change
                        const d = d_oj - d_n;
                        for (let i = 0; i < K; ++i) {
                            if (i !== n) deltaTD[i] += d;
                        }
                    }
                }
                // remember best swap for i;
                deltaTD.forEach((d, i) => {
                    if (d < DeltaTD[i]) {
                        DeltaTD[i] = d;
                        xs[i] = j;
                    }
                });
            }
        }
        let [min_idx, min_val] = min_index(DeltaTD);
        // stop if no improvements were found
        if (min_val >= 0) return true;
        // execute all improvements
        while (min_val < 0) {
            // swap roles of medoid m_i and non_medoid xs_i
            if (medoids.indexOf(xs[min_idx]) < 0) {
                medoids[min_idx] = xs[min_idx];
            }
            // disable the swap just performed
            DeltaTD[min_idx] = 0;
            // recompute TD for remaining swap candidates
            DeltaTD.forEach((d_j, j) => {
                if (d_j < 0) {
                    let sum = 0;
                    for (let o = 0; o < N; ++o) {
                        if (medoids.findIndex(m => m != j && m == o) >= 0) continue;
                        if (min_idx == j) continue;
                        if (cache[o].index_nearest === medoids[j])
                            sum += (Math.min(D.entry(o, j), cache[o].distance_second) - cache[o].distance_nearest);
                        else {
                            sum += (Math.min(D.entry(o, j) - cache[o].distance_nearest, 0));
                        }
                    }
                    if (sum < DeltaTD[j]) DeltaTD[j] = sum;
                }
            });
            [min_idx, min_val] = min_index(DeltaTD);
        }
        this._cluster_medoids = medoids;
        return false;
    }

    _nearest_medoid(j) {
        const row_j = j * this._N;
        const medoids = this._cluster_medoids;
        const D = this._distance_matrix.values;
        const dists_j = medoids.map((m, i) => [D[row_j + m], i]);
        const [nearest, second] = quickselect(dists_j, (m1, m2) => m1[0] - m2[0], 1);
        return {
            "distance_nearest": nearest[0],
            "index_nearest": nearest[1],
            "distance_second": second[0],
            "index_second": second[1],
        };
    }

    /**
     * Computes {@link K} clusters out of the {@link matrix}.
     * @param {Number} K - number of clusters.
     */
    init(K, cluster_medoids) {
        if (!K) K = this._K;
        if (!cluster_medoids) cluster_medoids = this._get_random_medoids(K);
        const max_iter = this._max_iter;
        let finish = false;
        let i = 0;
        do {
            finish = this._iteration();
        } while (!finish && ++i < max_iter)
        return this;
    }

    /**
     * Algorithm 3. FastPAM LAB: Linear Approximate BUILD initialization.
     * @param {number} K - number of clusters
     * 
     */
    _get_random_medoids(K) {
        const N = this._N;
        const D = this._distance_matrix.values;
        const randomizer = this._randomizer;
        const n = Math.min(N, 10 + Math.ceil(Math.sqrt(N)));

        // first medoid
        let m0 = -1;
        let TD = Infinity;
        let indices = Array.from({ length: N }, (_, i) => i);
        let S = randomizer.choice(indices, n);
        for (const x_j of S) {
            let TD_j = 0;
            const row_j = x_j * N;
            for (const x_o of S) {
                if (x_o !== x_j) TD_j += D[row_j + x_o];
            }
            if (TD_j < TD) {
                TD = TD_j; // smallest distance sum
                m0 = x_j;
            }
        }

        // other medoids
        const medoids = [m0];
        for (let i = 1; i < K; ++i) {
            let DeltaTD = Infinity;
            indices = indices.filter(d => d !== m0);
            S = randomizer.choice(indices, n);
            for (const x_j of S) {
                let deltaTD = 0;
                for (const x_o of S) {
                    if (x_o !== x_j) {
                        const row_o = x_o * N;
                        const d = D[row_o + x_j] - min(medoids.map(m => D[row_o + m]));
                        if (d < 0) deltaTD += d;
                    }
                }
                // best reduction
                if (deltaTD < DeltaTD) {
                    DeltaTD = deltaTD;
                    m0 = x_j;
                }
            }
            TD += DeltaTD;
            medoids.push(m0);
        }
        return medoids;
    }
}
