use crate::neural_group::NeuralGroup;
use crate::pattern_map::PatternMap;
use rand::Rng;
use rand::RngExt;

pub struct Repertoire {
    neural_groups: Vec<NeuralGroup>,
    mappings: Vec<PatternMap>,
    input_size: usize,
}

impl Repertoire {
    pub fn new(group_size: usize, group_count: usize, input_size: usize) -> Self {
        let mut neural_groups = Vec::with_capacity(group_count);
        let mut mappings = Vec::with_capacity(group_count);

        for _i in 0..group_count {
            neural_groups.push(NeuralGroup::new(group_size));
            mappings.push(PatternMap::new(group_size));
        }

        Self { 
            neural_groups,
            mappings,
            input_size,
        }
    }

    pub fn randomize<R: Rng>(&mut self, rng: &mut R) {
        for i in 0..self.neural_groups.len() {
            self.neural_groups[i].randomize(rng);
            self.mappings[i].randomize(rng, self.input_size);
        }
    }

    pub fn update_async_random<R: Rng>(&mut self, rng: &mut R) {
        let group_index = rng.random_range(0..self.neural_groups.len());
        self.neural_groups[group_index].update_async_random(rng);
    }

    pub fn set_input(&mut self, input: &[f64]) {
        for i in 0..self.neural_groups.len() {
            self.mappings[i].map_to(input, self.neural_groups[i].state_mut());
        }
    }

    pub fn get_energy(&self, energy_out: &mut [f64]) {
        for i in 0..self.neural_groups.len() {
            energy_out[i] = self.neural_groups[i].energy();
        }
    }
}
