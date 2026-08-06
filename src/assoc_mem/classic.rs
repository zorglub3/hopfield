use super::AssociativeMemory;
use crate::smatrix::SMatrix;
use rand::Rng;
use rand::RngExt;
use std::marker::PhantomData;

fn sign(x: f64) -> f64 {
    if x >= 0. {
        1.
    } else {
        -1.
    }
}

pub trait LearningRule {
    fn learn(weights: &mut SMatrix<f64>, state: &[f64], amount: f64);
}

pub struct HebbianLearning;

impl LearningRule for HebbianLearning {
    fn learn(weights: &mut SMatrix<f64>, state: &[f64], amount: f64) {
        let a = amount / (state.len() as f64);

        for r in 0..weights.rows_count() {
            for c in (r + 1)..weights.cols_count() {
                weights[(r, c)] += a * state[r] * state[c];
            }
        }
    }
}

pub struct OjaLearning;

impl LearningRule for OjaLearning {
    fn learn(weights: &mut SMatrix<f64>, state: &[f64], amount: f64) {
        for r in 0..weights.rows_count() {
            let y = state[r];
            for c in (r + 1)..weights.cols_count() {
                weights[(r, c)] += amount * y * (state[c] - y * weights[(r, c)]);
            }
        }
    }
}

pub struct StorkeyLearning;

impl LearningRule for StorkeyLearning {
    fn learn(weights: &mut SMatrix<f64>, state: &[f64], amount: f64) {
        let mut h = Vec::with_capacity(state.len());

        for i in 0..state.len() {
            h.push(weights.row_mul(i, state, 0.));
        }

        for r in 0..weights.rows_count() {
            for c in (r + 1)..weights.cols_count() {
                weights[(r, c)] +=
                    amount * (state[r] * state[c] - state[r] * h[c] - h[r] * state[c]);
            }
        }
    }
}

pub struct ClassicHopfieldNetwork<L: LearningRule> {
    weights: SMatrix<f64>,
    state: Vec<f64>,
    phantom: PhantomData<L>,
}

impl<L: LearningRule> AssociativeMemory for ClassicHopfieldNetwork<L> {
    fn new(size: usize) -> Self {
        Self {
            weights: SMatrix::new(size, 0.),
            state: vec![0.; size],
            phantom: PhantomData,
        }
    }

    fn state_size(&self) -> usize {
        self.state.len()
    }

    fn input_state(&self) -> &[f64] {
        &self.state
    }

    fn input_state_mut(&mut self) -> &mut [f64] {
        &mut self.state
    }

    fn energy(&self) -> f64 {
        let mut acc = 0.;

        for r in 0..self.state_size() {
            for c in 0..self.state_size() {
                acc -= self.weights[(r, c)] * self.state[r] * self.state[c];
            }
        }

        acc * 0.5
    }

    fn learn(&mut self, amount: f64) {
        L::learn(&mut self.weights, &self.state, amount);
    }

    fn update_async(&mut self, index: usize) {
        let new_state_value = self.weights.row_mul(index, &self.state, 0.);
        self.state[index] = sign(new_state_value);
    }

    fn update_sync(&mut self) {
        let mut new_state = vec![0.; self.state_size()];

        for (i, s) in new_state.iter_mut().enumerate() {
            *s = sign(self.weights.row_mul(i, &self.state, 0.));
        }

        self.state.copy_from_slice(&new_state);
    }

    fn update_async_prob<R: Rng>(&mut self, index: usize, beta: f64, rng: &mut R) {
        let h = self.weights.row_mul(index, &self.state, 0.);
        let p = 0.5 * (1. + (beta * h).tanh());

        self.state[index] = if rng.random_bool(p) {
            1.
        } else {
            -1.
        };
    }

    fn update_sync_prob<R: Rng>(&mut self, beta: f64, rng: &mut R) {
        let mut new_state = vec![0.; self.state_size()];

        for (i, s) in new_state.iter_mut().enumerate() {
            let h = self.weights.row_mul(i, &self.state, 0.);
            let p = 0.5 * (1. + (beta * h).tanh());
            *s = if rng.random_bool(p) { 1. } else { -1. };
        }
    }

    fn randomize<R: Rng>(&mut self, rng: &mut R) {
        for r in 0..self.state_size() {
            for c in r..self.state_size() {
                if c == r {
                    self.weights[(r, c)] = 0.;
                } else {
                    self.weights[(r, c)] = rng.random_range(-1. ..1.);
                }
            }
        }
    }
}
