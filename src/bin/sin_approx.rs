use std::f64::consts::PI;

use continous_nns::{RNN, Matrix};
use micrograd::Value;

fn main() {
    let mut rnn = RNN::random(1, 1, 50);
    train(&mut rnn);
    test(&rnn);
}

/// YES THIS RNN SUCKS
fn test(rnn: &RNN) {
    let mut hidden = Matrix::new(rnn.hid_size, 1);
    for i in 0..10 {
        let (nhid, out) = rnn.forward(hidden, vec![Value::new(i as f64/ 10.)].into());
        let eval = out.get(0,0).eval();
        hidden = nhid; 
        let viz = (eval*15.).abs().floor() as usize;
        for _ in 0..viz {
            print!("#");
        }
        println!();
    }
}

fn train(rnn: &mut RNN) {
    let train_steps = 100;
    let epoch_size = 30;
    let train_data = (0..train_steps).map(|_| {
        (0..epoch_size).map(|_| {
            let rand_len = rand::random::<u64>()%2+5;
            let rand_start = rand::random::<f64>()*PI;
            (0..rand_len).map(move |j| {
                let x = (j as f64) * 0.5 + rand_start;
                (x)
            }).collect::<Vec<_>>()
        }).collect::<Vec<_>>()
    }).collect::<Vec<_>>();

    let loss_speed = 0.003;

    for (epoch_ind, epoch) in train_data.into_iter().enumerate() {
        println!("Cur epoch: {}", epoch_ind);
        let mut loss = Value::new(0.);
        for (sequence_ind, sequence) in epoch.into_iter().enumerate() {
            println!("Cur seq: {}", sequence_ind);
            let mut hid = Matrix::new(rnn.hid_size, 1);
            for val in sequence {
                let (nhid, out) = rnn.forward(hid, vec![Value::new(val)].into());
                hid = nhid;
                assert!(out.rows == 1 && out.cols == 1);
                let out = out.get(0, 0);
                let y = Value::new(val.sin());
                let diff = out-y;
                loss = loss+diff.c()*diff;
            }
        }
        println!("Average loss: {}", loss.eval()/epoch_size as f64);
        let label = format!("epoch {}", epoch_ind);
        println!("Computing gradient...");
        loss.compute_gradient(&label);
        println!("Gradient computation finished");
        for par in rnn.params() {
            let nval = par.eval() - par.grad(&label) * loss_speed / (epoch_size as f64);
            *par = Value::new(nval);
        }
    }
}