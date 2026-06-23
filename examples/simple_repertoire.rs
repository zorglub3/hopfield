use hopfield::repertoire::Repertoire;
use hopfield::assoc_mem::classic::*;

const NEURAL_GROUP_COUNT: usize = 128;
const NEURAL_GROUP_SIZE: usize = 100;
const INPUT_SIZE: usize = 32;

fn main() {
    println!("Making repertoire");
    let mut repertoire: Repertoire<ClassicHopfieldNetwork<StorkeyLearning>> = 
        Repertoire::new(NEURAL_GROUP_SIZE, NEURAL_GROUP_COUNT, INPUT_SIZE);

    println!("Randomizing weights");
    let mut rng = rand::rng();
    repertoire.randomize(&mut rng);

    println!("Done");
}
