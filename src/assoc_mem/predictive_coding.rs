use crate::smatrix::SMatrix;
use rand::Rng;
use super::AssociativeMemory;

fn activation(v: f64) -> f64 {
    v.tanh()
}

#[allow(unused)]
pub struct PredictiveCoding {
    weights: SMatrix<f64>,
    state: Vec<f64>,
    error: Vec<f64>,
    prediction: Vec<f64>,
    target: Vec<f64>,
    mask: Vec<bool>,
}

impl AssociativeMemory for PredictiveCoding {
    fn new(size: usize) -> Self {
        Self {
            weights: SMatrix::new(size, 0.),
            state: vec![0.; size],
            error: vec![0.; size],
            prediction: vec![0.; size],
            target: vec![0.; size],
            mask: vec![false; size],
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
        todo!()
    }

    fn learn(&mut self, _amount: f64) {
        todo!()
    }

    fn update_async(&mut self, _index: usize) {
        todo!()
    }

    fn update_sync(&mut self) {
        self.update_predictions();
        self.update_errors();
        self.update_state();
    }

    fn update_async_prob<R: Rng>(&mut self, _index: usize, _beta: f64, _rng: &mut R) {
        todo!()
    }

    fn update_sync_prob<R: Rng>(&mut self, _beta: f64, _rng: &mut R) {
        todo!()
    }

    fn randomize<R: Rng>(&mut self, _rng: &mut R) {
        todo!()
    }
}

impl PredictiveCoding {
    pub fn set_target_and_mask(&mut self, target: &[f64], mask: &[bool]) {
        self.target.copy_from_slice(target);
        self.mask.copy_from_slice(mask);
    }

    pub fn update_errors(&mut self) {
        for (index, e) in self.error.iter_mut().enumerate() {
            if self.mask[index] {
                *e = self.target[index] - self.prediction[index];
            } else {
                *e = self.state[index] - self.prediction[index];
            }
        }
    }

    pub fn update_predictions(&mut self) {
        for (r, p) in self.prediction.iter_mut().enumerate() {
            *p = activation(self.weights.row_mul(r, &self.state, 0.));
        }
    }

    pub fn update_state(&mut self) {
        todo!()
    }
}
