use micrograd::Value;

use crate::{layer::{LayeredNetwork, TanhLayer}, Matrix};

pub struct RNN {
    w_hh: Matrix,
    w_xh: Matrix,
    w_hy: Matrix,
    b_h: Matrix,
    pub inp_size: usize,
    pub hid_size: usize,
    pub out_size: usize
}

impl RNN {
    pub fn random(inp_size: usize, out_size: usize, hid_size: usize) -> Self {
        let w_hh = Matrix::random(hid_size, hid_size);
        let w_xh = Matrix::random(hid_size, inp_size);
        let w_hy = Matrix::random(out_size, hid_size);
        let b_h = Matrix::random(hid_size, 1);

        Self {
            w_hh,
            w_xh,
            w_hy,
            b_h,
            inp_size,
            out_size,
            hid_size
        }
    }
    /// Returns (hidden, output) matrix in order
    pub fn forward(&self, hid_prev: Matrix, inp: Matrix) -> (Matrix, Matrix) {
        let hid = (self.w_hh.c()*hid_prev+self.w_xh.c()*inp + self.b_h.c()).map(|val, _, _| val.tanh());
        let out = self.w_hy.c()*hid.c();
        (hid, out)
    }

    pub fn params(&mut self) -> Vec<&mut Value> {
        self.w_hh.iter_mut().flatten().chain(self.w_hy.iter_mut().flatten()).chain(self.w_xh.iter_mut().flatten()).chain(self.b_h.iter_mut().flatten()).collect::<Vec<_>>()
    }
}