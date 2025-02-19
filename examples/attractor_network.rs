use hopfield::classic::*;
use hopfield::smatrix::SMatrix;
use hopfield::state::State;
use rand::Rng;

const STATE_SIZE: usize = 200;
const NETWORK_COUNT: usize = 20;
const PATTERN_AMOUNT: f64 = 0.5;
const NOISE_AMOUNT: f64 = 0.2;
const INITIAL_NOISE_AMOUNT: f64 = 0.1;
const STATE_DECAY: f64 = 0.5;
const LEARN_AMOUNT: f64 = 1. / 200.;
const LEARNING_NOISE_AMOUNT: f64 = 0.1;
const ITERATION_PRINT: usize = 100;

fn read_integer() -> Option<u32> {
    let stdin = std::io::stdin();
    let mut buffer = String::new();
    stdin.read_line(&mut buffer).ok()?;
    buffer.trim().parse::<u32>().ok()
}

fn main() {
    let mut weights: Vec<SMatrix<f64>> = Vec::new();
    let mut states: Vec<Vec<f64>> = Vec::new();
    let mut rng = rand::rng();

    for _i in 0 .. NETWORK_COUNT {
        let mut w = SMatrix::new(STATE_SIZE, 0.);
        initialize_weights(&mut w, &mut rng, INITIAL_NOISE_AMOUNT);
        let mut s = vec![0.; STATE_SIZE];
        s.add_noise(&mut rng, 1.0);

        weights.push(w);
        states.push(s);
    }

    let mut current_output: Vec<f64> = vec![0.; STATE_SIZE];

    let mut error_avg: f64 = 0.;
    let mut error_count: f64 = 0.;
    let mut iteration = 0;

    while let Some(input) = read_integer() {
        iteration += 1;

        let mut pattern: Vec<f64> = vec![0.; STATE_SIZE];
        let mut temp_state: Vec<f64> = vec![0.; STATE_SIZE];
        let zeros: Vec<f64> = vec![0.; STATE_SIZE];

        pattern.from_bits(STATE_SIZE, input);

        let mut min_error = f64::MAX;
        let mut min_net: Option<usize> = None;

        for i in 0 .. NETWORK_COUNT {
            states[i].decay(STATE_DECAY);
            states[i].add_pattern(&current_output, PATTERN_AMOUNT);
            states[i].add_noise(&mut rng, NOISE_AMOUNT);

            update_state_sync(&weights[i], &zeros, &states[i], &mut temp_state);

            let err = temp_state.error_norm(&pattern);

            states[i].copy_from(&temp_state);

            if err < min_error {
                min_error = err;
                min_net = Some(i);
            }
        }

        if let Some(winner) = min_net {
            let selected = rng.random_range(0 .. NETWORK_COUNT);

            temp_state.copy_from(&states[winner]);
            temp_state.add_noise(&mut rng, LEARNING_NOISE_AMOUNT);

            storkey_learn(&mut weights[selected], &temp_state, LEARN_AMOUNT);

            current_output.copy_from(&states[winner]);
        }

        error_avg += min_error;
        error_count += 1.;

        if iteration % ITERATION_PRINT == 0 {
            println!("{}", error_avg / error_count);
            error_avg = 0.;
            error_count = 0.;
        }
    }
}

