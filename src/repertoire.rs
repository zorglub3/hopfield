use crate::assoc_mem::AssociativeMemory;
use crate::pattern_map::PatternMap;
use crate::state::State;
use crate::smatrix::SMatrix;
use rand::Rng;
use rand::RngExt;

struct NeuralGroup<G: AssociativeMemory> {
    memory: G,
    mapping: PatternMap,
}

impl<G: AssociativeMemory> NeuralGroup<G> {
    fn new(size: usize) -> Self {
        Self {
            memory: G::new(size),
            mapping: PatternMap::new(size),
        }
    }

    fn randomize<R: Rng>(&mut self, input_size: usize, rng: &mut R) {
        self.mapping.randomize(rng, input_size);
        self.memory.randomize(rng);
    }

    fn set_input(&mut self, input_pattern: &[f64]) {
        self.mapping.map_to(input_pattern, self.memory.input_state_mut());
    }

    fn update_async_random<R: Rng>(&mut self, rng: &mut R) {
        self.memory.update_async_random(rng);
    }

    fn learn(&mut self, alpha: f64) {
        self.memory.learn(alpha);
    }

    fn energy(&self) -> f64 {
        self.memory.energy()
    }
}

pub struct Repertoire<G: AssociativeMemory> {
    neural_groups: Vec<NeuralGroup<G>>,
    input_size: usize,
    group_weights: SMatrix<f64>,
}

impl<G: AssociativeMemory> Repertoire<G> {
    pub fn new(group_size: usize, group_count: usize, input_size: usize) -> Self {
        let mut neural_groups = Vec::with_capacity(group_count);

        for _i in 0..group_count {
            neural_groups.push(NeuralGroup::new(group_size));
        }

        Self {
            neural_groups,
            input_size,
            group_weights: SMatrix::new(group_count, 0.),
        }
    }

    pub fn group_count(&self) -> usize {
        self.neural_groups.len()
    }

    pub fn neural_group(&self, index: usize) -> &G {
        &self.neural_groups[index].memory
    }

    pub fn neural_group_mut(&mut self, index: usize) -> &mut G {
        &mut self.neural_groups[index].memory
    }

    pub fn mapping(&self, index: usize) -> &PatternMap {
        &self.neural_groups[index].mapping
    }

    pub fn mapping_mut(&mut self, index: usize) -> &mut PatternMap {
        &mut self.neural_groups[index].mapping
    }

    pub fn randomize<R: Rng>(&mut self, rng: &mut R) {
        for neural_group in &mut self.neural_groups {
            neural_group.randomize(self.input_size, rng);
        }
        self.randomize_group_weights(rng);
    }

    pub fn update_async_random<R: Rng>(&mut self, rng: &mut R) {
        let group_index = rng.random_range(0..self.neural_groups.len());
        self.neural_groups[group_index].update_async_random(rng);
    }

    pub fn set_input(&mut self, input_pattern: &[f64]) {
        for neural_group in &mut self.neural_groups {
            neural_group.set_input(input_pattern);
        }
    }

    fn neural_group_energy(&self) -> Vec<f64> {
        let mut energy = Vec::with_capacity(self.neural_groups.len());

        for neural_group in &self.neural_groups {
            energy.push(neural_group.energy());
        }

        energy
    }

    pub fn learn_pattern(&mut self, pattern: &[f64], alpha: f64, sigma: f64) {
        self.set_input(pattern);
        let mut energy = self.neural_group_energy();
        energy.softmin_inplace(sigma);

        for (i, ng) in self.neural_groups.iter_mut().enumerate() {
            ng.learn(alpha * energy[i]);
        }
    }

    fn randomize_group_weights<R: Rng>(&mut self, rng: &mut R) {
        for r in 0..self.group_weights.rows_count() {
            for c in (r + 1)..self.group_weights.cols_count() {
                self.group_weights[(r, c)] = rng.random_range(-1. .. 1.);
            }
        }
    }
}
