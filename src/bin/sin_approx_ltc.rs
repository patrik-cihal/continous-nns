use std::f64::consts::PI;

use continous_nns::{LTC};
use micrograd::Value;

fn main() {
    let mut ltc = LTC::random(1, 1, 10);
    train(&mut ltc);
    test(&mut ltc);
}

fn test(ltc: &mut LTC) {
    ltc.set_inputs(vec![0.0.into()]);


    for i in 1..10 {
        let t = i as f64 * 0.5;
        ltc.set_inputs(vec![t.into()]);
        ltc.ode_solve_euler(0.5, 0.2);

        let out = ltc.output()[0].c();

        let eval = out.eval();
        let viz = (eval*15.).floor() as usize;
        for _ in 0..viz {
            print!("#");
        }
        println!();
    }
}

fn train(mut ltc: &mut LTC) {
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

    let loss_speed = 0.0001;

    for (epoch_ind, epoch) in train_data.into_iter().enumerate() {
        println!("Cur epoch: {}", epoch_ind);
        let mut loss = Value::new(0.);
        let ltc_init = ltc.clone();
        for (sequence_ind, sequence) in epoch.into_iter().enumerate() {
            println!("Cur seq: {}", sequence_ind);
            for t in sequence.into_iter() {
                ltc.set_inputs(vec![Value::new(t)]);
                ltc.ode_solve_euler(0.5, 0.1);

                let out = ltc.output()[0].c();
                let y = Value::new(t.sin()+1.);
                let diff = out-y;
                loss = loss+diff.c()*diff;
            }
            *ltc = ltc_init.clone();
        }
        println!("Average loss: {}", loss.eval()/epoch_size as f64);
        let label = format!("epoch {}", epoch_ind);
        println!("Computing gradient...");
        loss.compute_gradient(&label);
        println!("Gradient computation finished");
        for par in ltc.params() {
            let Some(grad) = par.grad(&label) else {
                continue;
            };
            let nval = par.eval() - grad * loss_speed / (epoch_size as f64);
            *par = Value::new(nval);
        }
    }
}