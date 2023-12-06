use micrograd::Value;

#[derive(Clone)]
struct Neuron {
    val: Value,
    time_constant: Value
}

impl Neuron {
    fn random() -> Self {
        Self {
            val: Value::new(rand::random::<f64>()*2.),
            time_constant: Value::new(rand::random::<f64>())
        }
    }
}

#[derive(Clone)]
struct Synapse {
    weight: Value,
    a: Value,
    from: usize,
    to: usize,
}

impl Synapse {
    fn random(from: usize, to: usize) -> Self {
        Self {
            weight: Value::new(rand::random::<f64>()*5.),
            a: Value::new(rand::random::<f64>() * 5. +1.),
            from,
            to
        }
    }
}

#[derive(Clone)]
pub struct LTC {
    sensory_in: Vec<Value>,
    neurons: Vec<Neuron>,
    synapses: Vec<Synapse>,
    cur_time: f64,
    output_size: usize,
    input_synapses: Vec<Synapse>
}

impl LTC {
    pub fn random(input_size: usize, output_size: usize, neuron_cnt: usize) -> Self {
        Self {
            sensory_in: (0..input_size).map(|_| Value::new(0.)).collect(),
            neurons: (0..neuron_cnt).map(|_| Neuron::random()).collect(),
            synapses: (0..neuron_cnt).map(|i| (0..neuron_cnt).map(move |j| Synapse::random(i, j))).flatten().collect(),
            cur_time: 0.,
            output_size,
            input_synapses: (0..neuron_cnt).map(|i| (0..input_size).map(move |j| Synapse::random(j, i))).flatten().collect::<Vec<_>>()
        }
    }
    pub fn set_inputs(&mut self, inputs: Vec<Value>) {
        self.sensory_in = inputs;
    }
    pub fn ode_solve_euler(&mut self, duration: f64, euler_step_size: f64) {
        let target_time = self.cur_time+duration;
        while self.cur_time < target_time {
            let step_size = (target_time-self.cur_time).min(euler_step_size);
            self.perform_euler_step(step_size);
        }
    }
    fn perform_euler_step(&mut self, step_size: f64) {
        let mut neuron_in = (0..self.neurons.len()).map(|_| Value::new(0.)).collect::<Vec<_>>();
        for synapse in &self.synapses {
            neuron_in[synapse.to] = neuron_in[synapse.to].c() + synapse.weight.c()*self.neurons[synapse.to].val.c().tanh()*(synapse.a.c()-self.neurons[synapse.to].val.c());
        }
        for synapse in &self.input_synapses {
            neuron_in[synapse.to] = neuron_in[synapse.to].c() + self.sensory_in[synapse.from].c().tanh()*(synapse.a.c()-self.neurons[synapse.to].val.c());
        }
        for (neuron, neuron_in) in self.neurons.iter_mut().zip(neuron_in) {
            let neuron_deriv = neuron.val.c()*(Value::new(-1.))*neuron.time_constant.c().powf(-1.) + neuron_in;
            neuron.val = neuron.val.c() + neuron_deriv*Value::new(step_size);
        }
        self.cur_time += step_size;
    }

    pub fn output(&self) -> Vec<Value> {
        self.neurons[0..self.output_size].iter().map(|n| n.val.clone()).collect::<Vec<_>>()
    }

    pub fn params(&mut self) -> Vec<&mut Value> {
        self.neurons.iter_mut().map(|n| &mut n.time_constant).chain(self.synapses.iter_mut().map(|syn| vec![&mut syn.a, &mut syn.weight]).flatten()).chain(self.input_synapses.iter_mut().map(|syn| vec![&mut syn.a, &mut syn.weight]).flatten()).collect()
    }
}