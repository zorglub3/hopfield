use crate::classic::*;
use crate::smatrix::SMatrix;
use rand::Rng;
use rand::RngExt;

pub struct NeuralGroup {
    weights: SMatrix<f64>,
    state: Vec<f64>,
}

impl NeuralGroup {
    pub fn new(size: usize) -> Self {
        Self {
            weights: SMatrix::new(size, 0.),
            state: vec![0.; size],
        }
    }

    pub fn randomize<R: Rng>(&mut self, rng: &mut R) {
        initialize_weights(&mut self.weights, rng, 1.);
    }

    pub fn set(&mut self, pattern: &[f64]) {
        debug_assert_eq!(pattern.len(), self.state.len());
        self.state.clone_from_slice(pattern);
    }

    pub fn set_from_boolean(&mut self, pattern: &[bool]) {
        let pattern_f64 = pattern.iter().map(|b| if *b { 1. } else { -1. }).collect::<Vec<_>>();
        self.set(&pattern_f64);
    }

    pub fn get(&self) -> &[f64] {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut [f64] {
        &mut self.state
    }

    pub fn energy(&self) -> f64 {
        energy(&self.weights, &self.state)
    }

    pub fn learn(&mut self, amount: f64) {
        storkey_learn(&mut self.weights, &self.state, amount);
    }

    pub fn update_async(&mut self, index: usize) {
        update_state_async(&self.weights, &mut self.state, index);
    }

    pub fn update_async_random<R: Rng>(&mut self, rng: &mut R) {
        let index = rng.random_range(0..self.state.len());
        self.update_async(index);
    }
}
