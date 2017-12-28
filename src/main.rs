extern crate rand;
use rand::distributions::{Normal, IndependentSample};

#[macro_use(array)]
extern crate ndarray;
use ndarray::{Array1, Array2};


#[derive(Debug)]
struct Network {
	sizes: Array1<usize>,
	weights: Array1<Array2<f64>>,
	biases: Array1<Array1<f64>>
}

impl Network {

	fn new(sizes: Array1<usize>) -> Network {

		let normal = Normal::new(0.0, 1.0);


		let (weights, biases) = {

		let random_bias = |_: usize| -> f64 {
			return normal.ind_sample(&mut rand::thread_rng())
		};

		let random_weight = |_: (usize, usize)| -> f64 {
			return normal.ind_sample(&mut rand::thread_rng())
		};


		let bias_layer = |i: usize| -> Array1<f64> {
			return Array1::from_shape_fn(sizes[i + 1], &random_bias)
		};

		let weight_layer = |i: usize| -> Array2<f64> {
			return Array2::from_shape_fn((sizes[i + 1], sizes[i]), &random_weight)
		};

		let weights: Array1<Array2<f64>> = Array1::from_shape_fn(sizes.len() - 1, weight_layer);
		let biases: Array1<Array1<f64>> = Array1::from_shape_fn(sizes.len() - 1, bias_layer);

		(weights, biases)
		};

		return Network {sizes: sizes,
						weights: weights,
						biases: biases}
	}
}

fn main() {
    let net = Network::new(array![2,3,1]);
    println!("My network : {:?}", net);
}
