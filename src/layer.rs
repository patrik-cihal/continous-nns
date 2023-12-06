use micrograd::Value;

use crate::matrix::Matrix;


pub struct TanhLayer {
    weights: Matrix,
    bias: Matrix
}

impl TanhLayer {
    pub fn random01(inp_size: usize, neuron_cnt: usize) -> Self {
        let weights = Matrix::random(neuron_cnt, inp_size);
        let bias = Matrix::random(neuron_cnt, 1);
        Self {
            bias,
            weights
        }
    }
    pub fn forward(&self, x: Matrix) -> Matrix {
        ((self.weights.c()*x) + self.bias.c()).map(|val, _, _| val.tanh())
    }
    pub fn params(&mut self) -> Vec<&mut Value> {
        self.weights.iter_mut().flatten().chain(self.bias.iter_mut().flatten()).collect::<Vec<_>>()
    }
}

pub struct LayeredNetwork {
    layers: Vec<TanhLayer>,
}

impl LayeredNetwork {
    pub fn random01(shape: Vec<usize>) -> Self {
        let layers = shape.windows(2).map(|x| {
            assert!(x.len() == 2);
            TanhLayer::random01(x[0], x[1])
        }).collect::<Vec<_>>();

        Self {
            layers
        }
    }

    pub fn forward(&self, mut x: Matrix) -> Matrix {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }

    pub fn params(&mut self) -> Vec<&mut Value> {
        self.layers.iter_mut().map(|layer| layer.params()).flatten().collect::<Vec<_>>()
    }
}