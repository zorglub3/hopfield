//! Some utility methods that are useful for manipulating the state of various Hopfield
//! models. The state is represented with a slice.

use rand::Rng;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::{BitAnd, Shr};

pub trait State {
    fn error_norm(&self, pattern: &[f64]) -> f64;
    fn copy_from(&mut self, pattern: &[f64]);
    fn decay(&mut self, d: f64);
    fn add_pattern(&mut self, pattern: &[f64], amount: f64);
    fn add_noise(&mut self, rng: &mut impl Rng, amount: f64);
    fn from_bits<S: BitAnd<Output = S> + Shr<Output = S> + From<u8> + PartialEq + Copy>(
        &mut self,
        count: usize,
        bits: S,
    );
    fn from_bits_with_mask<S: BitAnd<Output = S> + Shr<Output = S> + From<u8> + PartialEq + Copy>(
        &mut self,
        count: usize,
        bits: S,
        mask: S,
    );
    fn softmax(&mut self);
}

impl<T: Deref<Target = [f64]> + DerefMut<Target = [f64]>> State for T {
    fn error_norm(&self, pattern: &[f64]) -> f64 {
        let mut acc = 0.;
        let mut pat_acc = 0.;

        for i in 0..self.len().min(pattern.len()) {
            acc += self[i] * pattern[i];
            pat_acc += pattern[i].abs();
        }

        if pat_acc < f64::EPSILON {
            0.
        } else {
            1. - acc / pat_acc
        }
    }

    fn copy_from(&mut self, pattern: &[f64]) {
        for i in 0..self.len().min(pattern.len()) {
            self[i] = pattern[i];
        }
    }

    fn decay(&mut self, d: f64) {
        for i in 0..self.len() {
            self[i] *= d;
        }
    }

    fn add_pattern(&mut self, pattern: &[f64], amount: f64) {
        for i in 0..self.len().min(pattern.len()) {
            self[i] += pattern[i] * amount;
        }
    }

    fn add_noise(&mut self, rng: &mut impl Rng, amount: f64) {
        for i in 0..self.len() {
            self[i] += rng.random_range(-amount..amount);
        }
    }

    fn from_bits<S: BitAnd<Output = S> + Shr<Output = S> + From<u8> + PartialEq + Copy>(
        &mut self,
        count: usize,
        mut bits: S,
    ) {
        for i in 0..count.min(self.len()) {
            if bits & 1.into() == 1.into() {
                self[i] = 1.;
            } else {
                self[i] = -1.;
            }

            bits = bits >> 1.into();
        }
    }

    fn from_bits_with_mask<
        S: BitAnd<Output = S> + Shr<Output = S> + From<u8> + PartialEq + Copy,
    >(
        &mut self,
        count: usize,
        mut bits: S,
        mut mask: S,
    ) {
        for i in 0..count.min(self.len()) {
            if mask & 1.into() == 1.into() {
                if bits & 1.into() == 1.into() {
                    self[i] = 1.;
                } else {
                    self[i] = -1.;
                }
            } else {
                self[i] = 0.;
            }

            bits = bits >> 1.into();
            mask = mask >> 1.into();
        }
    }

    fn softmax(&mut self) {
        if self.len() > 0 {
            let mut acc = 0.;

            for i in 0..self.len() {
                let v = self[i].exp();
                acc += v;
                self[i] = v;
            }

            if acc > f64::EPSILON {
                for i in 0..self.len() {
                    self[i] /= acc;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn from_bits_works() {
        let v: u32 = 0b0011;
        let mut u = vec![0.0; 8];

        u.from_bits(16, v);

        assert_eq!(u[0], 1.);
        assert_eq!(u[1], 1.);
        assert_eq!(u[2], -1.);
        assert_eq!(u[3], -1.);
    }

    #[test]
    fn from_bits_with_mask_works() {
        let v: u32 = 0b1101;
        let m: u32 = 0b1011;
        let mut u = vec![0.0; 8];

        u.from_bits_with_mask(16, v, m);

        assert_eq!(u[0], 1.);
        assert_eq!(u[1], -1.);
        assert_eq!(u[2], 0.);
        assert_eq!(u[3], 1.);
    }

    #[test]
    fn decay_decreases_value() {
        let mut v = vec![10.0_f64; 8];

        v.decay(0.5);

        for i in 0..8 {
            assert!(v[i] < 10.0);
        }
    }

    #[test]
    fn add_pattern() {
        let mut v = vec![1.0_f64; 8];
        let p = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        v.add_pattern(&p, 0.5);

        for i in 0..8 {
            assert_eq!(v[i], 1.0 + ((i as f64) + 1.0) * 0.5);
        }
    }

    #[test]
    fn add_noise_is_bounded() {
        let mut v = vec![1.0_f64; 10000];
        let mut rng = rand::rng();

        v.add_noise(&mut rng, 1.0);

        for i in 0..8 {
            assert!(v[i] <= 2.0);
            assert!(v[i] >= 0.0);
        }
    }
}
