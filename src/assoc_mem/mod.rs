pub mod classic;
pub mod modern;
pub mod predictive_coding;

use rand::Rng;
use rand::RngExt;

pub trait AssociativeMemory {
    fn new(size: usize) -> Self;
    fn state_size(&self) -> usize;
    fn state(&self) -> &[f64];
    fn state_mut(&mut self) -> &mut [f64];
    fn energy(&self) -> f64;
    fn learn(&mut self, amount: f64);
    fn update_async(&mut self, index: usize);
    fn update_sync(&mut self);
    fn randomize<R: Rng>(&mut self, rng: &mut R);

    fn set_boolean_state(&mut self, pattern: &[bool]) {
        let pattern = pattern
            .iter()
            .map(|b| if *b { 1. } else { -1. })
            .collect::<Vec<f64>>();
        self.state_mut().copy_from_slice(&pattern);
    }

    fn update_async_random<R: Rng>(&mut self, rng: &mut R) {
        let index = rng.random_range(0..self.state_size());
        self.update_async(index);
    }

    fn randomize_state<R: Rng>(&mut self, rng: &mut R) {
        for s in self.state_mut() {
            *s = if rng.random_bool(0.5) { 1. } else { -1. };
        }
    }

    fn temperature_flux<R: Rng>(&mut self, _temperature: f64) {
        todo!()
    }
}
