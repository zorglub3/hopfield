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

pub struct PCN {
    weights: Vec<DMatrix<f64>>,
    neurons: Vec<Vec<f64>>,
    predictions: Vec<Vec<f64>>,
    errors: Vec<Vec<f64>>,
    layers: usize,
}

impl PCN {
    #[allow(dead_code)]
    pub fn pp_state(&self) {
        for l in 0..self.layers {
            println!("Layer {}", l);
            println!(" - neurons: {:?}", self.neurons[l]);
            println!(" - predictions: {:?}", self.predictions[l]);
            println!(" - errors: {:?}", self.errors[l]);
        }
    }

    #[allow(dead_code)]
    pub fn pp_weights(&self) {
        for l in 0..(self.layers - 1) {
            println!("Layer {} weights", l);
            self.weights[l].pp();
        }
    }

    pub fn new(layers: usize, n: usize, d: usize) -> Self {
        debug_assert!(layers > 1);
        debug_assert!(n > 0);
        debug_assert!(d > 1);

        let mut weights = Vec::new();
        weights.push(DMatrix::new(d, n, 0.));
        for _i in 0..(layers - 2) {
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
        for j in 0..self.neurons[0].len() {
            if sensor_values[j].abs() <= f64::EPSILON {
                self.errors[0][j] = self.neurons[0][j] - self.predictions[0][j];
            } else {
                self.errors[0][j] = sensor_values[j] - self.predictions[0][j];
            }
        }

        for i in 1..self.layers {
            for j in 0..self.neurons[i].len() {
                self.errors[i][j] = self.neurons[i][j] - self.predictions[i][j];
            }
        }
    }

    pub fn global_error(&self) -> f64 {
        let mut acc = 0.;

        for i in 0..self.layers {
            for j in 0..self.errors[i].len() {
                let e = self.errors[i][j];
                acc += e * e;
            }
        }

        acc * 0.5
    }

    pub fn inference_sensor_step(&mut self, sensor_mask: &[f64], gamma: f64) {
        for i in 0..self.neurons[0].len() {
            if sensor_mask[i].abs() <= f64::EPSILON {
                self.neurons[0][i] -= gamma * self.errors[0][i];
            }
        }
    }

    pub fn inference_step(&mut self, gamma: f64) {
        for i in 1..self.layers {
            let mut ad = self.neurons[i].clone();
            activation_diff(&mut ad);

            let mut et = vec![0.; self.neurons[i].len()];
            self.weights[i - 1].trans_mul_vec(&self.errors[i - 1], &mut et);

            for j in 0..self.neurons[i].len() {
                self.neurons[i][j] += gamma * (-self.errors[i][j] + ad[j] * et[j]);
            }
        }
    }

    pub fn inference(
        &mut self,
        memory_pattern: &[f64],
        sensor_pattern: &[f64],
        gamma: f64,
        steps: usize,
    ) {
        self.memory_mut().copy_from(memory_pattern);
        self.sensors_mut().copy_from(sensor_pattern);

        for _i in 0..steps {
            self.prediction(memory_pattern);
            self.error(sensor_pattern);
            self.inference_step(gamma);
        }
    }

    pub fn inference_with_sensors(
        &mut self,
        memory_pattern: &[f64],
        sensor_pattern: &[f64],
        gamma: f64,
        steps: usize,
    ) {
        self.memory_mut().copy_from(memory_pattern);
        self.sensors_mut().copy_from(sensor_pattern);

        for _i in 0..steps {
            self.prediction(memory_pattern);
            self.error(sensor_pattern);
            self.inference_sensor_step(sensor_pattern, gamma);
            self.inference_step(gamma);
        }
    }

    pub fn learn(&mut self, alpha: f64) {
        for i in 0..(self.layers - 1) {
            for r in 0..self.weights[i].rows() {
                for c in 0..self.weights[i].cols() {
                    self.weights[i][(r, c)] +=
                        alpha * self.errors[i][r] * self.neurons[i][c].tanh();
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

    #[test]
    fn can_learn_a_pattern() {
        const SENSORS: usize = 16;
        const MEMORY: usize = 8;

        let mut pcn = PCN::new(3, MEMORY, SENSORS);
        let mut s = vec![0.; SENSORS];
        let mut m = vec![0.; MEMORY];

        for i in 0..SENSORS {
            s[i] = if i % 2 == 0 { 1. } else { -1. };
        }
        for i in 0..MEMORY {
            m[i] = if i % 2 == 0 { 1. } else { -1. };
        }

        const T: usize = 1000;
        const U: usize = 10;
        const GAMMA: f64 = 0.1;
        const ALPHA: f64 = 0.1;

        for _i in 0..U {
            pcn.inference(&m, &s, GAMMA, T);
            pcn.learn(ALPHA);
        }

        let mut pattern = vec![0.; SENSORS];
        for i in 0..(SENSORS / 2) {
            pattern[i] = if i % 2 == 0 { 1. } else { -1. };
        }

        pcn.memory_mut().copy_from(&m);
        pcn.sensors_mut().copy_from(&pattern);

        pcn.inference_with_sensors(&m, &pattern, GAMMA, T);

        for i in 0..SENSORS {
            assert!((pcn.sensors()[i] - s[i]).abs() < 0.1);
        }

        assert!(pcn.global_error() < 0.1);
    }

    #[test]
    fn will_not_recall_pattern_it_hasnt_learned() {
        const SENSORS: usize = 16;
        const MEMORY: usize = 8;

        let mut pcn = PCN::new(2, MEMORY, SENSORS);
        let mut s = vec![0.; SENSORS];
        let mut m = vec![0.; MEMORY];

        for i in 0..SENSORS {
            s[i] = if i < (SENSORS / 2) { 1. } else { -1. };
        }
        for i in 0..MEMORY {
            m[i] = if i % 2 == 0 { 1. } else { -1. };
        }

        const T: usize = 1000;
        const U: usize = 10;
        const GAMMA: f64 = 0.1;
        const ALPHA: f64 = 0.1;

        for _i in 0..U {
            pcn.inference(&m, &s, GAMMA, T);
            pcn.learn(ALPHA);
        }

        let mut pattern = vec![0.; SENSORS];
        for i in 0..(SENSORS / 2) {
            pattern[i] = if i % 2 == 0 { 1. } else { -1. };
        }

        pcn.inference_with_sensors(&m, &pattern, GAMMA, T);

        pcn.memory_mut().copy_from(&m);
        pcn.error(&s);

        assert!(pcn.global_error() >= 5.0);
    }
}
