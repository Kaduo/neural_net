extern crate rand;
use rand::distributions::{Normal, IndependentSample};

extern crate itertools;

#[macro_use(array)]
extern crate ndarray;
use ndarray::{Array1, Array2, Zip};
use ndarray::linalg::general_mat_vec_mul;

fn sigmoid(z: f64) -> f64 {
	return 1.0/(1.0+(-z).exp())
}

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

		let random_bias = |_i: usize| -> f64 {
			return normal.ind_sample(&mut rand::thread_rng())
		};

		let random_weight = |_i: (usize, usize)| -> f64 {
			return normal.ind_sample(&mut rand::thread_rng())
		};


		let bias_layer = |i: usize| -> Array1<f64> {
			return Array1::from_shape_fn(sizes[i + 1], &random_bias)
		};

		let weight_layer = |i: usize| -> Array2<f64> {
			return Array2::from_shape_fn((sizes[i + 1], sizes[i]), &random_weight)
		};

		(Array1::from_shape_fn(sizes.len() - 1, weight_layer), Array1::from_shape_fn(sizes.len() - 1, bias_layer))

		};

		return Network {sizes: sizes,
						weights: weights,
						biases: biases}
	}

	fn feed_forward(&self, input: &Array1<f64>) -> Array1<f64> {
		let mut output: Array1<f64> = input.clone();
		for (b, w) in itertools::zip(&self.biases, &self.weights) {
			let mut temp: Array1<f64> = Array1::zeros(output.dim());
			general_mat_vec_mul(1.0, &w, &output, 0.0, &mut temp);
			output = temp + b;
		}
		return output;
	}
}

fn main() {
    let mut net = Network::new(array![2,3,1]);
    println!("My biases : {:?}", net.biases);
    Zip::from(&mut net.biases[0]).apply(|x| {*x = sigmoid(*x)});
    println!("My new biases : {:?}", net.biases);
}
