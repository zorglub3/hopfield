pub mod classic;
pub mod modern;
pub mod predictive_coding;
pub mod rbm;

use rand::Rng;
use rand::RngExt;

pub trait AssociativeMemory {
    fn new(size: usize) -> Self;
    fn state_size(&self) -> usize;
    fn input_state(&self) -> &[f64];
    fn input_state_mut(&mut self) -> &mut [f64];
    fn energy(&self) -> f64;
    fn learn(&mut self, amount: f64);
    fn update_async(&mut self, index: usize);
    fn update_sync(&mut self);
    fn randomize<R: Rng>(&mut self, rng: &mut R);

    fn update_async_prob<R: Rng>(&mut self, index: usize, beta: f64, rng: &mut R);
    fn update_sync_prob<R: Rng>(&mut self, beta: f64, rng: &mut R);

    fn set_boolean_input(&mut self, pattern: &[bool]) {
        let pattern = pattern
            .iter()
            .map(|b| if *b { 1. } else { -1. })
            .collect::<Vec<f64>>();
        self.input_state_mut().copy_from_slice(&pattern);
    }

    fn update_async_random<R: Rng>(&mut self, rng: &mut R) {
        let index = rng.random_range(0..self.state_size());
        self.update_async(index);
    }

    fn update_async_random_prob<R: Rng>(&mut self, beta: f64, rng: &mut R) {
        let index = rng.random_range(0..self.state_size());
        self.update_async_prob(index, beta, rng);
    }

    fn randomize_input<R: Rng>(&mut self, rng: &mut R) {
        for s in self.input_state_mut() {
            *s = if rng.random_bool(0.5) { 1. } else { -1. };
        }
    }
}
