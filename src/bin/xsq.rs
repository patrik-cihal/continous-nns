use continous_nns::*;

pub fn main() {
    let mut network = LayeredNetwork::random01(vec![1, 8, 8, 1]);

    train(&mut network);

    let check = |v: f64| {
        network.forward(vec![Value::from(v)].into()).get(0, 0).eval()
    };
    dbg!(check(2.));
    dbg!(check(0.5));
    dbg!(check(0.3));

    dbg!(test(&network));
}

fn train(network: &mut LayeredNetwork) {
    let train_steps = 1000;
    let training_data = (0..train_steps).into_iter().map(|_| {
        let x = rand::random::<f64>();
        (x, x*x)
    }).collect::<Vec<_>>();
    let loss_speed = 0.1;
    for (step, (x, y)) in training_data.into_iter().enumerate() {
        let output = network.forward(vec![Value::new(x)].into());
        let output = output.into_iter().flatten().collect::<Vec<_>>();
        assert!(output.len() == 1);
        let output = output[0].c();
        let diff = Value::new(y) - output;
        let loss = diff.c()*diff;
        let label = format!("loss {}", step);
        loss.compute_gradient(&label);
        for param in network.params() {
            let new_val = param.eval()-param.grad(&label)*loss_speed;
            assert!(!new_val.is_nan());
            *param = Value::new(new_val);
        }
    }
}

fn test(network: &LayeredNetwork) -> f64 {
    let testing_steps = 1000;
    let testing_data = (0..testing_steps).into_iter().map(|_| {
        let x = rand::random::<f64>();
        (x, x*x)
    }).collect::<Vec<_>>();
    let mut av_loss = 0.;
    for (step, (x, y)) in testing_data.into_iter().enumerate() {
        let output = network.forward(vec![Value::new(x)].into());
        let output = output.into_iter().flatten().collect::<Vec<_>>();
        assert!(output.len() == 1);
        let output = output[0].c();
        let diff = Value::new(y) - output;
        let loss = diff.c()*diff;
        assert!(loss.eval().is_sign_positive());
        av_loss+=loss.eval();
    }
    av_loss/testing_steps as f64
}