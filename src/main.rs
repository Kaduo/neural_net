extern crate rand;
use rand::distributions::{Normal, IndependentSample};


#[derive(Debug)]
struct Network {
	sizes: Vec<u32>,
	weights: Vec<Vec<Vec<f64>>>,
	biases: Vec<Vec<f64>>
}

impl Network {
	fn new(sizes: Vec<u32>) -> Network {

		let mut rng = rand::thread_rng();
		let normal = Normal::new(0.0, 1.0);

		let mut weights: Vec<Vec<Vec<f64>>> = Vec::new();
		let mut biases: Vec<Vec<f64>> = Vec::new();

		let mut prev_size = sizes[0];
		for size in &sizes[1..] {
			let mut layer_biases: Vec<f64> = Vec::new();
			let mut layer_weights: Vec<Vec<f64>> = Vec::new();
			for _ in 0..*size {
				layer_biases.push(normal.ind_sample(&mut rng));
				let mut neuron_weights: Vec<f64> = Vec::new();
				for _ in 0..prev_size {
					neuron_weights.push(normal.ind_sample(&mut rng));
				}
				layer_weights.push(neuron_weights);
			}
			prev_size = *size;
			biases.push(layer_biases);
			weights.push(layer_weights);
		}
		return Network {sizes: sizes,
						weights: weights,
						biases: biases}
	}
}

fn main() {
    let net = Network::new(vec![2,3,1]);
    println!("My network : {:?}", net);
}
