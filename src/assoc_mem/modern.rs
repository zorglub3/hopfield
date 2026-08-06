use super::AssociativeMemory;
use crate::state::State;
use num::pow;
use rand::Rng;

fn sign(x: f64) -> f64 {
    if x >= 0. {
        1.
    } else {
        -1.
    }
}

fn f(x: f64, n: usize) -> f64 {
    debug_assert!(n >= 1);

    pow(x, n)
}

pub struct ModernHopfieldNetwork<const N: usize> {
    state: Vec<f64>,
    patterns: Vec<Vec<f64>>,
}

impl<const N: usize> AssociativeMemory for ModernHopfieldNetwork<N> {
    fn new(size: usize) -> Self {
        Self {
            state: vec![0.; size],
            patterns: Vec::new(),
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
        for pattern in &self.patterns {
            acc -= f(self.state.dot(pattern), N);
        }
        acc
    }

    fn learn(&mut self, _amount: f64) {
        todo!()
    }

    fn update_async(&mut self, index: usize) {
        let mut acc = 0.;

        for pattern in &self.patterns {
            let mut acc2 = 0.;

            for (j, p) in pattern.iter().enumerate() {
                if j != index {
                    acc2 += p * self.state[j];
                }
            }

            acc += f(acc2 + pattern[index], N) - f(acc2 - pattern[index], N);
        }

        self.state[index] = sign(acc);
    }

    fn update_sync(&mut self) {
        let mut pattern_select = Option::default();
        let mut pattern_energy = f64::MAX;

        for i in 0..self.patterns.len() {
            let energy = f(self.patterns[i].dot(&self.state), N);
            if energy < pattern_energy {
                pattern_select = Some(i);
                pattern_energy = energy;
            }
        }

        if let Some(index) = pattern_select {
            self.state.copy_from_slice(&self.patterns[index]);
        }
    }

    fn update_async_prob<R: Rng>(&mut self, _index: usize, _beta: f64, _rng: &mut R) {
        unimplemented!();
    }

    fn update_sync_prob<R: Rng>(&mut self, _beta: f64, _rng: &mut R) {
        unimplemented!();
    }

    fn randomize<R: Rng>(&mut self, _rng: &mut R) {
        // do nothing
    }
}
