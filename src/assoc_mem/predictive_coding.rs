use crate::smatrix::SMatrix;
use rand::Rng;
// use rand::RngExt;
use super::AssociativeMemory;

#[allow(unused)]
pub struct PredictiveCoding {
    weights: SMatrix<f64>,
    state: Vec<f64>,
}

impl AssociativeMemory for PredictiveCoding {
    fn new(size: usize) -> Self {
        Self {
            weights: SMatrix::new(size, 0.),
            state: vec![0.; size],
        }
    }

    fn state_size(&self) -> usize {
        self.state.len()
    }

    fn state(&self) -> &[f64] {
        &self.state
    }

    fn state_mut(&mut self) -> &mut [f64] {
        &mut self.state
    }

    fn energy(&self) -> f64 {
        todo!()
    }

    fn learn(&mut self, _amount: f64) {
        todo!()
    }

    fn update_async(&mut self, _index: usize) {
        todo!()
    }

    fn update_sync(&mut self) {
        todo!()
    }

    fn randomize<R: Rng>(&mut self, _rng: &mut R) {
        todo!()
    }
}
