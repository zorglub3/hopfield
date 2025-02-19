use crate::dmatrix::DMatrix;
use crate::state::State;

fn activation(vs: &mut [f64]) {
    for i in 0..vs.len() {
        vs[i] = vs[i].tanh();
    }
}

fn activation_diff(vs: &mut [f64]) {
    for i in 0..vs.len() {
        let t = vs[i].tanh();
        vs[i] = 1. - t * t;
    }
}

struct PCN {
    weights: Vec<DMatrix<f64>>,
    neurons: Vec<Vec<f64>>,
    predictions: Vec<Vec<f64>>,
    errors: Vec<Vec<f64>>,
    layers: usize,
}

impl PCN {
    pub fn new(layers: usize, n: usize, d: usize) -> Self {
        debug_assert!(layers > 1);
        debug_assert!(n > 0);
        debug_assert!(d > 1);

        let mut weights = Vec::new();
        weights.push(DMatrix::new(d, n, 0.));
        for _i in 0..(layers - 1) {
            weights.push(DMatrix::new(n, n, 0.));
        }

        let mut neurons = Vec::new();
        let mut predictions = Vec::new();
        let mut errors = Vec::new();

        neurons.push(vec![0.; d]);
        predictions.push(vec![0.; d]);
        errors.push(vec![0.; d]);

        for _i in 0..(layers - 1) {
            neurons.push(vec![0.; n]);
            predictions.push(vec![0.; n]);
            errors.push(vec![0.; n]);
        }

        Self {
            weights,
            neurons,
            predictions,
            errors,
            layers,
        }
    }

    pub fn memory(&self) -> &[f64] {
        &self.neurons[self.layers - 1]
    }

    pub fn memory_mut(&mut self) -> &mut [f64] {
        &mut self.neurons[self.layers - 1]
    }

    pub fn sensors(&self) -> &[f64] {
        &self.neurons[0]
    }

    pub fn sensors_mut(&mut self) -> &mut [f64] {
        &mut self.neurons[0]
    }

    pub fn prediction(&mut self, memory_values: &[f64]) {
        self.predictions[self.layers - 1].copy_from(memory_values);

        for i in (0..(self.layers - 1)).rev() {
            let mut temp_vec = self.neurons[i + 1].clone();
            activation(&mut temp_vec);
            self.weights[i].mul_vec(&temp_vec, &mut self.predictions[i]);
        }
    }

    pub fn error(&mut self, sensor_values: &[f64]) {
        for i in 0..self.layers {
            for j in 0..self.neurons[i].len() {
                self.errors[i][j] = self.neurons[i][j] - self.predictions[i][j];
            }
        }
    }

    pub fn inference(&mut self, gamma: f64) {
        for i in 1..self.layers {
            let mut ad = self.neurons[i].clone();
            activation_diff(&mut ad);

            let mut et = vec![0.; self.neurons[i].len()];
            self.weights[i].trans_mul_vec(&self.errors[i - 1], &mut et);

            for j in 0..self.neurons[i].len() {
                self.neurons[i][j] += gamma * (-self.errors[i][j] + ad[j] * et[j]);
            }
        }
    }

    pub fn learn(&mut self, alpha: f64) {
        for i in 0..(self.layers - 1) {
            for r in 0..self.weights[i].rows() {
                for c in 0..self.weights[i].cols() {
                    self.weights[i][(r, c)] +=
                        alpha * self.errors[i][r] * self.neurons[i + 1][c].tanh();
                }
            }
        }

        for i in 0..self.memory().len() {
            self.neurons[self.layers - 1][i] -= alpha * self.errors[self.layers - 1][i];
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::state::State;

    #[test]
    fn can_set_sensor_value() {
        let mut pcn = PCN::new(2, 4, 8);
        let s = vec![1., 1., -1., -1., 1., 1., -1., -1.];

        pcn.sensors_mut().copy_from(&s);

        for i in 0..8 {
            assert_eq!(pcn.sensors()[i], s[i]);
        }
    }
}
