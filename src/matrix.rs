use micrograd::Value;
use std::ops::{Mul, Add};

#[derive(Clone)]
pub struct Matrix {
    data: Vec<Vec<Value>>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = (0..rows).map(|_| (0..cols).map(|_| Value::new(0.)).collect::<Vec<_>>()).collect::<Vec<_>>();
        Self {
            data,
            rows, cols
        }
    }
    pub fn random(rows: usize, cols: usize) -> Self {
        let data = (0..rows).map(|_| (0..cols).map(|_| Value::new(rand::random::<f64>()*2.-1.)).collect::<Vec<_>>()).collect::<Vec<_>>();
        Self {
            data,
            rows, cols
        }
    }
    pub fn c(&self) -> Self {
        self.clone()
    }
    pub fn set(&mut self, row: usize, col: usize, val: Value) {
        self.data[row][col] = val;
    }
    pub fn get(&self, row: usize, col: usize) -> Value {
        self.data[row][col].clone()
    }
    pub fn apply<F: Fn(Value, usize, usize) -> Value>(&mut self, f: F) {
        self.data.iter_mut().enumerate().for_each(|(row_ind, row)| {
            row.iter_mut().enumerate().for_each(|(col_ind, val)| {
                *val = (f)(val.clone(), row_ind, col_ind);
            });
        });
    }
    pub fn map<F: Fn(Value, usize, usize) -> Value>(mut self, f: F) -> Matrix {
        self.apply(f);
        self
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Vec<micrograd::Value>> {
        self.data.iter_mut()
    }
    pub fn into_iter(self) -> std::vec::IntoIter<Vec<micrograd::Value>> {
        self.data.into_iter()
    }
}

impl From<Vec<Value>> for Matrix {
    fn from(values: Vec<Value>) -> Self {
        let data = values.into_iter().map(|val| vec![val]).collect::<Vec<_>>();
        Self {
            rows: data.len(),
            cols: 1,
            data
        }
    }
}

impl Mul<Matrix> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Self::Output {
        assert!(self.cols == rhs.rows);
        let mut res = Matrix::random(self.rows, rhs.cols);
        for row in 0..self.rows {
            for rhs_col in 0..rhs.cols {
                let mut sum = Value::new(0.);
                for col in 0..self.cols {
                    sum = sum + self.get(row, col)*rhs.get(col, rhs_col);
                }
                res.set(row, rhs_col, sum);
            }
        }
        res
    }
}

impl Add<Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Self::Output {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols);
        let mut res = Matrix::random(self.rows, self.cols);
        res.apply(|_, row, col| {
            self.get(row, col)+rhs.get(row, col) 
        });
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use micrograd::Value;

    // ... existing test_matrix_new test ...

    #[test]
    fn test_matrix_random() {
        let rows = 4;
        let cols = 3;
        let matrix = Matrix::random(rows, cols);

        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.cols, cols);
        for row in matrix.data.iter() {
            assert_eq!(row.len(), cols);
            for val in row.iter() {
                assert!(val.eval() >= -1.0 && val.eval() <= 1.0); // Random values should be in the range [-1.0, 1.0]
            }
        }
    }

    #[test]
    fn test_set_get() {
        let mut matrix = Matrix::random(2, 2);
        let value = Value::new(5.0);
        matrix.set(1, 1, value.clone());

        assert_eq!(matrix.get(1, 1).eval(), value.eval());
    }

    #[test]
    fn test_clone() {
        let mut matrix = Matrix::random(2, 2);
        matrix.set(0, 0, Value::new(2.0));
        let clone = matrix.c();

        assert_eq!(clone.get(0, 0).eval(), 2.0);
    }

    #[test]
    fn test_apply() {
        let mut matrix = Matrix::random(2, 2);
        matrix.apply(|val, _, _| Value::new(val.eval() + 1.0));

        for row in matrix.data.iter() {
            for val in row.iter() {
                assert_eq!(val.eval(), 1.0);
            }
        }
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Matrix::from(vec![Value::new(1.0), Value::new(2.0)]); // 2x1 matrix
        let b = Matrix { data: vec![vec![Value::new(3.0), Value::new(4.0)]], rows: 1, cols: 2 }; // 1x2 matrix
    
        let result = a * b;
        assert_eq!(result.get(0, 0).eval(), 3.0);  // 1.0 * 3.0
        assert_eq!(result.get(0, 1).eval(), 4.0);  // 1.0 * 4.0
        assert_eq!(result.get(1, 0).eval(), 6.0);  // 2.0 * 3.0
        assert_eq!(result.get(1, 1).eval(), 8.0);  // 2.0 * 4.0
    }

    #[test]
    fn test_matrix_addition() {
        let a = Matrix::from(vec![Value::new(1.0), Value::new(2.0)]);
        let b = Matrix::from(vec![Value::new(3.0), Value::new(4.0)]);

        let result = a + b;
        assert_eq!(result.get(0, 0).eval(), 4.0);
        assert_eq!(result.get(1, 0).eval(), 6.0);
    }

    #[test]
    #[should_panic]
    fn test_incompatible_multiplication() {
        let a = Matrix::random(2, 3);
        let b = Matrix::random(4, 2);
        let _ = a * b; // Should panic due to incompatible sizes
    }

    #[test]
    #[should_panic]
    fn test_incompatible_addition() {
        let a = Matrix::random(2, 2);
        let b = Matrix::random(3, 3);
        let _ = a + b; // Should panic due to incompatible sizes
    }

    // ... other tests as needed ...
}