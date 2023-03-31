import { distance_matrix, Matrix } from "../matrix/index.js";
import { euclidean_squared } from "../metrics/index.js";
import { DR } from "./DR.js";

/**
 * @class
 * @alias TSNE
 * @extends DR
 */
export class TSNE extends DR {
    /**
     *
     * @constructor
     * @memberof module:dimensionality_reduction
     * @alias TSNE
     * @param {Matrix} X - the high-dimensional data.
     * @param {Object} parameters - Object containing parameterization of the DR method.
     * @param {Number} [parameters.perplexity = 50] - perplexity.
     * @param {Number} [parameters.epsilon = 10] - learning parameter.
     * @param {Number} [parameters.d = 2] - the dimensionality of the projection.
     * @param {Function|"precomputed"} [parameters.metric = euclidean_squared] - the metric which defines the distance between two points.
     * @param {Number} [parameters.seed = 1212] - the seed for the random number generator.
     * @returns {TSNE}
     */
    constructor(X, parameters) {
        super(X, { perplexity: 50, epsilon: 10, d: 2, metric: euclidean_squared, seed: 1212 }, parameters);
        [this._N, this._D] = this.X.shape;
        this._iter = 0;
        return this;
    }

    /**
     *
     * @returns {TSNE}
     */
    init() {
        // init
        const { d: dim, metric, perplexity } = this._parameters;
        const randomizer = this._randomizer;
        const Htarget = Math.log(perplexity);
        const N = this._N;
        const X = this.X;
        const Delta = metric === "precomputed" ? X : distance_matrix(X, metric);
        const P = new Matrix(N, N, 0);

        this.Y = new Matrix(N, dim, () => randomizer.gauss_random() * 1e-4);
        this._ystep = new Matrix(N, dim, 0);
        this._gains = new Matrix(N, dim, 1);

        // search for fitting sigma
        const tol = 1e-4;
        const maxtries = 50;
        for (let i = 0; i < N; ++i) {
            const dist_i = Delta.row(i);
            const prow = P.row(i);
            let betamin = -Infinity;
            let betamax = Infinity;
            let beta = 1;
            let cnt = maxtries;
            let done = false;
            let psum, dp_sum;

            while (!done && cnt--) {
                // compute entropy and kernel row with beta precision
                psum = dp_sum = 0;
                for (let j = 0; j < N; ++j) {
                    const dist = dist_i[j];
                    const pj = (i !== j) ? Math.exp(-dist * beta) : 1e-9;
                    dp_sum += beta * dist * pj;
                    prow[j] = pj;
                    psum += pj;
                }
                // compute entropy
                const H = Math.log(psum) + (dp_sum / psum);
                if (H > Htarget) {
                    betamin = beta;
                    beta = betamax === Infinity ? 2 * beta : 0.5 * (beta + betamax);
                } else {
                    betamax = beta;
                    beta = betamin === -Infinity ? 0.5 * beta : 0.5 * (beta + betamin);
                }
                done = Math.abs(H - Htarget) < tol;
            }
            // normalize row
            const N2 = 1.0 / (2 * N * psum);
            for (let j = 0; j < N; ++j) {
                prow[j] *= N2;
            }
        }

        // compute probabilities
        for (let i = 0; i < N; ++i) {
            const P_i = P.row(i);
            for (let j = i; j < N; ++j) {
                P.set_entry(j, i, P_i[j] += P.entry(j, i));
            }
        }
        this._P = P;
        return this;
    }

    /**
     *
     * @param {Number} [iterations=500] - Number of iterations.
     * @returns {Matrix|Number[][]} the projection.
     */
    transform(iterations = 500) {
        this.check_init();
        for (let i = 0; i < iterations; ++i) {
            this.next();
        }
        return this.projection;
    }

    /**
     *
     * @param {Number} [iterations=500] - number of iterations.
     * @yields {Matrix|Number[][]} - the projection.
     */
    *generator(iterations = 500) {
        this.check_init();
        for (let i = 0; i < iterations; ++i) {
            this.next();
            yield this.projection;
        }
        return this.projection;
    }

    /**
     * performs a optimization step
     * @private
     * @returns {Matrix}
     */
    next() {
        const { d: dim, epsilon} = this._parameters;
        const ystep = this._ystep.values;
        const gains = this._gains.values;
        const iter = ++this._iter;
        const P = this._P;
        const N = this._N;
        const Y = this.Y;

        //calc cost gradient;
        const pmul = iter < 100 ? 4 : 1;
        const momval = iter < 250 ? 0.5 : 0.8;

        // compute Q dist (unnormalized)
        let d, dsum, qsum = 0;
        const Q = new Matrix(N, N, 0);
        for (let i = 0; i < N; ++i) {
            const Q_i = Q.row(i);
            const Y_i = Y.row(i);
            for (let j = i + 1; j < N; ++j) {
                for (dsum = 0, d = 0; d < dim; ++d) {
                    const dhere = Y_i[d] - Y.entry(j, d);
                    dsum += dhere * dhere;
                }
                const qu = 1 / (1 + dsum);
                Q.set_entry(j, i, Q_i[j] = qu);
                qsum += 2 * qu;
            }
        }

        const Y_val = Y.values;
        const grad = new Matrix(N, dim, 0);
        for (let i = 0; i < N; ++i) {
            const P_i = P.row(i);
            const Q_i = Q.row(i);
            const Y_i = Y.row(i);
            const g_i = grad.row(i);
            for (let j = 0; j < N; ++j) {
                if (i !== j) {
                    const qu = Q_i[j];
                    const premult = 4 * (pmul * P_i[j] - (qu / qsum)) * qu;
                    for (d = 0; d < dim; ++d) {
                        g_i[d] += premult * (Y_i[d] - Y.entry(j, d));
                    }
                }
            }
        }

        // perform gradient step
        const g_val = grad.values;
        const ymean = new Float64Array(dim);
        for (let i = 0, cnt = N; cnt--;) {
            for (d = 0; d < dim; ++d, ++i) {
                const gid = g_val[i];
                const sid = ystep[i];

                const newgain = Math.max(gid * sid < 0.0 ? gains[i] + 0.2 : gains[i] * 0.8, 0.01);
                gains[i] = newgain;

                const newsid = momval * sid - epsilon * newgain * gid;
                ystep[i] = newsid;

                ymean[d] += (Y_val[i] += newsid);
            }
        }

        // center Y around mean
        for (d = 0; d < dim; ++d) ymean[d] /= N;
        for (let i = 0, cnt = N; cnt--;) {
            for (d = 0; d < dim;) Y_val[i++] -= ymean[d++];
        }

        return this.Y;
    }
}
