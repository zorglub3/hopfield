use crate::smatrix::SMatrix;
use rand::Rng;

fn activation(v: f64, bias: f64) -> f64 {
    if v >= bias {
        1.
    } else {
        -1.
    }
}

pub fn initialize_weights<R: Rng>(weights: &mut SMatrix<f64>, rng: &mut R, amount: f64) {
    for r in 0..weights.rows() {
        for c in r..weights.cols() {
            if c == r {
                weights[(r, c)] = 0.;
            } else {
                weights[(r, c)] = rng.random_range(-amount .. amount);
            }
        }
    }
}

pub fn update_state_sync(
    weights: &SMatrix<f64>,
    bias: &[f64],
    input_state: &[f64],
    output_state: &mut [f64],
) {
    let l = weights
        .rows()
        .min(input_state.len())
        .min(output_state.len());

    for i in 0..l {
        output_state[i] = activation(weights.row_mul(i, input_state, 0.), bias[i]);
    }
}

pub fn update_state_async(
    weights: &SMatrix<f64>, 
    bias: &[f64], 
    state: &mut [f64], 
    index: usize,
) {
    debug_assert!(index < state.len());
    debug_assert!(index < weights.rows());

    let new_state_value = activation(weights.row_mul(index, state, 0.), bias[index]);

    state[index] = new_state_value;
}

pub fn energy(weights: &SMatrix<f64>, bias: &[f64], state: &[f64]) -> f64 {
    let mut acc = 0.;

    for r in 0..state.len() {
        for c in 0..state.len() {
            acc -= weights[(r, c)] * state[r] * state[c];
        }

        acc -= bias[r] * state[r];
    }

    acc
}

pub fn hebb_learn(weights: &mut SMatrix<f64>, pattern: &[f64]) {
    let n_inv = 1. / (pattern.len() as f64);

    for r in 0 .. weights.rows() {
        for c in (r + 1) .. weights.cols() {
            weights[(r, c)] += n_inv * pattern[r] * pattern[c];
        }
    }
}

pub fn storkey_learn(weights: &mut SMatrix<f64>, pattern: &[f64], amount: f64) {
    let mut h = Vec::with_capacity(pattern.len());

    for i in 0 .. pattern.len() {
        h.push(weights.row_mul(i, pattern, 0.));
    }

    for r in 0 .. weights.rows() {
        for c in (r + 1) .. weights.cols() {
            weights[(r, c)] +=
                amount * (pattern[r] * pattern[c] - pattern[r] * h[c] - h[r] * pattern[c]);
        }
    }
}
