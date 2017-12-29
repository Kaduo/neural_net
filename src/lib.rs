extern crate itertools;
use itertools::zip;

extern crate nalgebra;
use nalgebra as na;
use na::{DMatrix, DVector};


fn sigmoid(z: f64) -> f64 {
	return 1.0/(1.0 + (-z).exp());
}

#[derive(Debug)]
pub struct Network {
	sizes: Vec<usize>,
	weights: Vec<DMatrix<f64>>,
	biases: Vec<DVector<f64>>,
	nb_layers: usize
}

impl Network {

	pub fn new(sizes: Vec<usize>) -> Network {

		let nb_layers = sizes.len();
		let mut weights: Vec<DMatrix<f64>> = Vec::with_capacity(nb_layers - 1);
		let mut biases: Vec<DVector<f64>> = Vec::with_capacity(nb_layers - 1);

		for layer in 1..nb_layers {
			biases.push(DVector::new_random(sizes[layer]));
			weights.push(DMatrix::new_random(sizes[layer], sizes[layer - 1]));
		}

		return Network {sizes: sizes,
						weights: weights,
						biases: biases,
						nb_layers: nb_layers}
	}

	fn feed_forward(&self, input: &DVector<f64>) -> DVector<f64> {
		let mut output: DVector<f64> = input.clone();
		for (w, b) in zip(&self.weights, &self.biases) {
			output = w*output + b;
			for x in &mut output {
				*x = sigmoid(*x);
			}
		}
		return output
	}

	fn SGD(&mut self, training_data: &Vec<(DVector<f64>, DVector<f64>)>,
			nb_epochs: usize, mini_batch_size: usize, learning_step: f64) {

		unimplemented!();
	}
}