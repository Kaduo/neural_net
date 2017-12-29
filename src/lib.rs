extern crate rand;
use rand::{thread_rng, Rng};

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

	fn backprop(&self, input: &DVector<f64>, expected_result: &DVector<f64>)
				-> (Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
		unimplemented!();
	}

	fn update_mini_batch(&mut self,
						mini_batch: &[(DVector<f64>, DVector<f64>)],
						learning_step: f64) {

		let mut nabla_biases: Vec<DVector<f64>> = Vec::with_capacity(self.nb_layers - 1);
		let mut nabla_weights: Vec<DMatrix<f64>> = Vec::with_capacity(self.nb_layers - 1);

		for layer in 1..self.nb_layers {
			nabla_biases.push(DVector::from_element(self.sizes[layer], 0.0));
			nabla_weights.push(DMatrix::from_element(self.sizes[layer], self.sizes[layer - 1], 0.0));
		}

		for ref tuple in mini_batch {

			let input = &tuple.0;
			let expected_result = &tuple.1;
			let (delta_nabla_b, delta_nabla_w) = self.backprop(&input, &expected_result);

			for layer in 1..self.nb_layers {
				nabla_biases[layer] += &delta_nabla_b[layer];
				nabla_weights[layer] += &delta_nabla_w[layer];
			}
		}

		for layer in 1..self.nb_layers {

			let normalized_step = learning_step/(mini_batch.len() as f64);
			self.biases[layer] = &self.biases[layer] - normalized_step*&nabla_biases[layer];

			self.weights[layer] = &self.weights[layer] - normalized_step*&nabla_weights[layer];
		}

		unimplemented!();
	}

	fn stochastic_gradient_descent(&mut self,
									training_data: &mut Vec<(DVector<f64>, DVector<f64>)>,
									nb_epochs: usize, mini_batch_size: usize, learning_step: f64) {

		let mut rng = thread_rng();
		let n = training_data.len();

		for _ in 0..nb_epochs {

			rng.shuffle(training_data);
			let mut mini_batches: Vec<&[(DVector<f64>, DVector<f64>)]> = Vec::new();

			let mut i = 0;
			while i + mini_batch_size < n {
				mini_batches.push(&training_data[i..(i + mini_batch_size)]);
				i += mini_batch_size;
			}

			for mini_batch in mini_batches.iter() {
				self.update_mini_batch(mini_batch, learning_step);
			}
		}
		unimplemented!();
	}
}