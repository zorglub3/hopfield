use crate::dmatrix::DMatrix;
use super::AssociativeMemory;
use rand::Rng;
use rand::RngExt;

fn g(v: f64) -> f64 {
    1. / (1. + (-v).exp())
}

fn activation(v: f64) -> f64 {
    if v >= 0. {
        1.
    } else {
        -1.
    }
}

pub struct RestrictedBoltzmannMachine {
    visible: Vec<f64>,
    hidden: Vec<f64>,
    weights: DMatrix<f64>,
}

impl AssociativeMemory for RestrictedBoltzmannMachine {
    fn new(size: usize) -> Self {
        Self::with_hidden(size, size)
    }

    fn state_size(&self) -> usize {
        self.visible.len()
    }

    fn input_state(&self) -> &[f64] {
        &self.visible
    }

    fn input_state_mut(&mut self) -> &mut [f64] {
        &mut self.visible
    }

    fn energy(&self) -> f64 {
        let mut acc = 0.;

        for (r, v) in self.visible.iter().enumerate() {
            for (c, h) in self.hidden.iter().enumerate() {
                acc -= self.weights[(r, c)] * v * h;
            }
        }

        acc
    }

    fn learn(&mut self, amount: f64) {
        let target_visible = self.visible.clone();
        let target_hidden = self.hidden.clone();

        self.update_hidden();
        self.update_visible();

        for r in 0..self.weights.rows() {
            for c in 0..self.weights.cols() {
                let delta =
                    target_visible[r] * target_hidden[c] - self.visible[r] * self.hidden[c];
                self.weights[(r, c)] += amount * delta;
            }
        }
    }

    fn update_async(&mut self, index: usize) {
        if index < self.visible.len() {
            let s = self.weights.mul_row_vec(&self.hidden, index);
            let v = if g(s) >= 0.5 { 1. } else { -1. };

            self.visible[index] = v;
        } else {
            let s = self.weights.mul_col_vec(&self.visible, index - self.visible.len());
            let v = if g(s) >= 0.5 { 1. } else { -1. };

            self.hidden[index - self.visible.len()] = v;
        }
    }

    fn update_sync(&mut self) {
        for (c, h) in self.hidden.iter_mut().enumerate() {
            *h = activation(self.weights.mul_col_vec(&self.visible, c));
        }

        for (r, v) in self.visible.iter_mut().enumerate() {
            *v = activation(self.weights.mul_row_vec(&self.hidden, r));
        }
    }

    fn update_async_prob<R: Rng>(&mut self, index: usize, beta: f64, rng: &mut R) {
        if index < self.visible.len() {
            let s = self.weights.mul_row_vec(&self.hidden, index);
            let q = g(beta * s);

            self.visible[index] = if rng.random_bool(q) { 1. } else { -1. };
        } else {
            let s = self.weights.mul_col_vec(&self.visible, index - self.visible.len());
            let q = g(beta * s);

            self.hidden[index - self.visible.len()] = if rng.random_bool(q) { 1. } else { -1. };
        }
    }

    fn update_sync_prob<R: Rng>(&mut self, beta: f64, rng: &mut R) {
        for (c, h) in self.hidden.iter_mut().enumerate() {
            let s = self.weights.mul_col_vec(&self.visible, c);
            let q = g(beta * s);

            *h = if rng.random_bool(beta * q) { 1. } else { -1. };
        }

        for (r, v) in self.visible.iter_mut().enumerate() {
            let s = self.weights.mul_row_vec(&self.hidden, r);
            let q = g(beta * s);

            *v = if rng.random_bool(beta * q) { 1. } else { -1. };
        }
    }

    fn randomize<R: Rng>(&mut self, rng: &mut R) {
        for r in 0..self.weights.rows() {
            for c in 0..self.weights.cols() {
                self.weights[(r, c)] = rng.random_range(-1. .. 1.);
            }
        }

        for h in self.hidden.iter_mut() {
            *h = if rng.random_bool(0.5) { 1. } else { -1. };
        }
    }
}

impl RestrictedBoltzmannMachine {
    pub fn with_hidden(visible_size: usize, hidden_size: usize) -> Self {
        Self {
            visible: vec![0.; visible_size],
            hidden: vec![0.; hidden_size],
            weights: DMatrix::new(visible_size, hidden_size, 0.),
        }
    }

    pub fn visible(&self) -> &[f64] {
        &self.visible
    }

    pub fn visible_mut(&mut self) -> &mut [f64] {
        &mut self.visible
    }

    pub fn hidden(&self) -> &[f64] {
        &self.hidden
    }

    pub fn hidden_mut(&mut self) -> &mut [f64] {
        &mut self.hidden
    }

    pub fn update_hidden(&mut self) {
        for (c, h) in self.hidden.iter_mut().enumerate() {
            *h = activation(self.weights.mul_col_vec(&self.visible, c));
        }
    }

    pub fn update_visible(&mut self) {
        for (r, v) in self.visible.iter_mut().enumerate() {
            *v = activation(self.weights.mul_row_vec(&self.hidden, r));
        }
    }
}
