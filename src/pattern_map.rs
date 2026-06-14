use rand::Rng;
use rand::seq::IteratorRandom;

pub struct PatternMap {
    mapping: Vec<usize>,
}

impl PatternMap {
    pub fn new(size: usize) -> Self {
        Self {
            mapping: vec![0; size],
        }
    }

    pub fn new_random<R: Rng>(size: usize, rng: &mut R, input_pattern_size: usize) -> Self {
        let mapping = (0..input_pattern_size).sample(rng, size);

        Self { mapping }
    }

    pub fn randomize<R: Rng>(&mut self, rng: &mut R, input_pattern_size: usize) {
        let mapping = (0..input_pattern_size).sample(rng, input_pattern_size);

        self.mapping = mapping;
    }

    pub fn map_to<T: Clone>(&self, input: &[T], target: &mut [T]) {
        for i in 0..self.mapping.len() {
            target[i] = input[self.mapping[i]].clone();
        }
    }
}
