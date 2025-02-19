use crate::dmatrix::DMatrix;
use crate::state::State;

fn activation(state: &mut [f64]) {
    for i in 0 .. state.len() {
        if state[i] >= 0. {
            state[i] = 1.;
        } else {
            state[i] = -1.;
        }
    }
}

pub fn update_state_sync(
    mat: &DMatrix<f64>,
    input_state: &[f64],
    output_state: &mut [f64],
) {
    let mut temp_vec1 = vec![0.; mat.rows()];
    let mut temp_vec2 = vec![0.; mat.rows()];

    mat.mul_vec(input_state, &mut temp_vec1);
    temp_vec2.softmax(&temp_vec1);
    mat.trans_mul_vec(&temp_vec2, output_state);
    activation(output_state);
}

pub fn learn(mat: &mut DMatrix<f64>, pattern: &[f64]) {
    debug_assert_eq!(mat.cols(), pattern.len());

    mat.add_row(pattern);
}

fn lse(beta: f64, x: &[f64]) -> f64 {
    let mut acc = 0.;

    for v in x {
        acc += (beta * v).exp();
    }

    acc.ln() / beta
}

pub fn energy(mat: &DMatrix<f64>, pattern: &[f64]) -> f64 {
    debug_assert_eq!(mat.cols(), pattern.len());

    let mut temp_vec = vec![0.; mat.rows()];
    mat.trans_mul_vec(pattern, &mut temp_vec);

    -lse(1., &temp_vec).exp()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn learned_patterns_are_added() {
        let mut m = DMatrix::new(1, 4, 0.);
        let p1: Vec<f64> = vec![1.; 4];
        let p2: Vec<f64> = vec![2.; 4];

        learn(&mut m, &p1);
        learn(&mut m, &p2);

        for c in 0 .. 4 { assert_eq!(m[(0, c)], 0.); }
        for c in 0 .. 4 { assert_eq!(m[(1, c)], 1.); }
        for c in 0 .. 4 { assert_eq!(m[(2, c)], 2.); }
    }

    #[test]
    fn can_recall_pattern() {
        let mut m = DMatrix::new(1, 8, 0.);
        let p: Vec<f64> = vec![1., 1., -1., -1., 1., 1., -1., -1.];
        let a: Vec<f64> = vec![1., 0., -1.,  0., 0., 1.,  0.,  0.];
        let mut output: Vec<f64> = vec![0.; 8];

        learn(&mut m, &p);

        update_state_sync(&m, &a, &mut output);

        println!("output: {:?}", output);
        for i in 0 .. 8 {
            assert_eq!(output[i], p[i]);
        }
    }
}

