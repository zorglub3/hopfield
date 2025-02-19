use std::fmt::Debug;
use std::ops::{AddAssign, Index, IndexMut, Mul};

pub struct SMatrix<T> {
    rows: usize,
    data: Vec<T>,
}

impl<T: Debug> SMatrix<T> {
    pub fn pp(&self) {
        for r in 0..self.rows {
            for c in 0..self.rows {
                print!("{:?} ", self[(r, c)]);
            }
            println!("");
        }
    }
}

impl<T> SMatrix<T> {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.rows
    }
}

impl<T: Default + Clone> SMatrix<T> {
    pub fn from_fn0<F>(rows: usize, f: &mut F) -> Self
    where
        F: FnMut() -> T,
    {
        let size = rows * (rows + 1) / 2;
        let mut data = Vec::with_capacity(size);

        for _i in 0..size {
            data.push(f());
        }

        Self { rows, data }
    }

    pub fn from_fn2<F>(rows: usize, f: &mut F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let size = rows * (rows + 1) / 2;
        let mut data = vec![T::default(); size];

        for c in 0..rows {
            for r in 0..=c {
                data[(2 * rows - r - 1) * r / 2 + c] = f(r, c);
            }
        }

        Self { rows, data }
    }
}

impl<T: Clone> SMatrix<T> {
    pub fn new(rows: usize, default_value: T) -> Self {
        let size = rows * (rows + 1) / 2;

        Self {
            rows,
            data: vec![default_value; size],
        }
    }
}

impl<T> SMatrix<T> {
    // this function has been taken from
    // https://stackoverflow.com/questions/3187957/how-to-store-a-symmetric-matrix
    fn row_col_to_index(&self, r: usize, c: usize) -> usize {
        if r <= c {
            (2 * self.rows - r - 1) * r / 2 + c
        } else {
            (2 * self.rows - c - 1) * c / 2 + r
        }
    }
}

impl<T: AddAssign + Clone> SMatrix<T> {
    pub fn add_matrix(&mut self, mat: &SMatrix<T>) {
        debug_assert_eq!(self.rows, mat.rows);

        for i in 0..self.data.len() {
            self.data[i] += mat.data[i].clone()
        }
    }
}

impl<T: Mul<Output = T> + AddAssign + Copy> SMatrix<T> {
    pub fn row_mul(&self, row: usize, vec: &[T], init: T) -> T {
        let mut acc = init;

        for i in 0..(self.rows.min(vec.len())) {
            acc += self[(row, i)] * vec[i];
        }

        acc
    }
}

impl<T> Index<(usize, usize)> for SMatrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &T {
        let idx = self.row_col_to_index(index.0, index.1);
        &self.data[idx]
    }
}

impl<T> IndexMut<(usize, usize)> for SMatrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        let idx = self.row_col_to_index(index.0, index.1);
        &mut self.data[idx]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn get_the_right_row() {
        let mat: SMatrix<usize> = SMatrix::from_fn2(8, &mut |r, _c| r);

        for r in 0..8 {
            for c in 0..8 {
                if r <= c {
                    assert_eq!(mat[(r, c)], r);
                } else {
                    assert_eq!(mat[(r, c)], c);
                }
            }
        }
    }

    #[test]
    fn row_mul_zero_matrix() {
        let mat: SMatrix<i32> = SMatrix::from_fn0(8, &mut || 0);
        let v: Vec<i32> = vec![0; 8];

        for r in 0..8 {
            assert_eq!(mat.row_mul(r, &v, 0), 0);
        }
    }

    #[test]
    fn random_matrix_is_symmetric() {
        use rand::Rng;

        let mut rng = rand::rng();
        let mat: SMatrix<f64> = SMatrix::from_fn0(8, &mut || rng.random_range(-1. ..1.));

        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(mat[(r, c)], mat[(c, r)]);
            }
        }
    }

    #[test]
    fn row_mul_nonzero() {
        let mat: SMatrix<i32> = SMatrix::from_fn2(8, &mut |r, c| (r + c) as i32);
        let v: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7];

        for i in 0..8 {
            let expect = 140 + (i as i32) * 28;
            assert_eq!(mat.row_mul(i, &v, 0), expect);
        }
    }

    #[test]
    fn add_zero_matrix() {
        let mut m1: SMatrix<i32> = SMatrix::from_fn2(8, &mut |r, c| (r + c) as i32);
        let m2: SMatrix<i32> = SMatrix::new(8, 0);

        m1.add_matrix(&m2);

        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(m1[(r, c)], (r + c) as i32);
            }
        }
    }

    #[test]
    fn add_nonzero_matrix() {
        let mut m1: SMatrix<i32> = SMatrix::from_fn2(8, &mut |r, c| (r + c) as i32);
        let m2: SMatrix<i32> = SMatrix::from_fn2(8, &mut |r, c| (2 * r + 2 * c) as i32);

        m1.add_matrix(&m2);

        for r in 0..8 {
            for c in 0..8 {
                let expected = (r as i32) * 3 + (c as i32) * 3;
                assert_eq!(m1[(r, c)], expected);
            }
        }
    }
}
