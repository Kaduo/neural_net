extern crate rand;
use rand::{thread_rng, Rng};

extern crate itertools;
use itertools::zip;

extern crate nalgebra;
use nalgebra as na;
use na::{DMatrix, DVector};

pub mod load_mnist;

fn sigmoid(z: f64) -> f64 {
	return 1.0/(1.0 + (-z).exp())
}

fn sigmoid_prime(z: f64) -> f64 {
	return sigmoid(z)*(1.0 - sigmoid(z))
}

fn cost_derivative(output_activations: &DVector<f64>,
					expected_output: &DVector<f64>) -> DVector<f64> {

	return output_activations - expected_output
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

	fn backprop(&self, input: &DVector<f64>, expected_output: &DVector<f64>)
				-> (Vec<DVector<f64>>, Vec<DMatrix<f64>>) {



		let n = self.nb_layers - 1;

		let mut nabla_biases: Vec<Option<DVector<f64>>> = vec![None; n];
		let mut nabla_weights: Vec<Option<DMatrix<f64>>> = vec![None; n];

		let mut activation = input.clone();
		let mut activations: Vec<DVector<f64>> = Vec::with_capacity(self.nb_layers);
		activations.push(activation.clone());

		let mut z_layers: Vec<DVector<f64>> = Vec::with_capacity(self.nb_layers);
		for (w, b) in zip(&self.weights, &self.biases) {
			let z = w*&activation + b;
			z_layers.push(z.clone());
			activation = z.clone();
			for x in &mut activation {
				*x = sigmoid(*x);
			}
			activations.push(activation.clone());
		}


		let mut z_last_prime: DVector<f64> = DVector::from_element(self.sizes[n], 0.0);

		for i in 0..self.sizes[n] {
			z_last_prime[i] = sigmoid(z_layers[n-1][i]);
		}
		
		let mut delta = cost_derivative(&activations[n], expected_output).component_mul(&z_last_prime);

		nabla_biases[n-1] = Some(delta.clone());
		nabla_weights[n-1] = Some(delta.clone() * activations[n-1].transpose());

		for l in 2..(n+1) {
			let z = z_layers[n-l].clone();
			let mut z_prime: DVector<f64> = DVector::from_element(self.sizes[n+1-l], 0.0);
			for i in 0..self.sizes[n+1-l] {
				z_prime[i] = sigmoid_prime(z[i]);
			}
			delta = (self.weights[n+1-l].transpose()*delta).component_mul(&z_prime);
			nabla_biases[n-l] = Some(delta.clone());
			nabla_weights[n-l] = Some(delta.clone() * activations[n-l].transpose());
		}

		let mut nabla_biases_final: Vec<DVector<f64>> = Vec::with_capacity(n);
		let mut nabla_weights_final: Vec<DMatrix<f64>> = Vec::with_capacity(n);

		for nabla_bias in nabla_biases {
			match nabla_bias {
				Some(x) => nabla_biases_final.push(x),
				None => panic!("Not every nabla_bias was computed !"),
			}
		}

		for nabla_weight in nabla_weights {
			match nabla_weight {
				Some(x) => nabla_weights_final.push(x),
				None => panic!("Not every nabla_weight was computed !"),
			}
		}

		return (nabla_biases_final, nabla_weights_final)

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
			let expected_output = &tuple.1;
			let (delta_nabla_b, delta_nabla_w) = self.backprop(&input, &expected_output);

			for layer in 0..(self.nb_layers - 1) {

				nabla_biases[layer] += &delta_nabla_b[layer];
				nabla_weights[layer] += &delta_nabla_w[layer];
			}
		}

		for layer in 0..(self.nb_layers - 1) {

			let normalized_step = learning_step/(mini_batch.len() as f64);
			self.biases[layer] = &self.biases[layer] - normalized_step*&nabla_biases[layer];

			self.weights[layer] = &self.weights[layer] - normalized_step*&nabla_weights[layer];

		}

	}

	pub fn stochastic_gradient_descent(&mut self,
									training_data: &mut Vec<(DVector<f64>, DVector<f64>)>,
									nb_epochs: usize, mini_batch_size: usize, learning_step: f64) {

		println!("Begining stochastic gradient descent");
		let mut rng = thread_rng();
		let n = training_data.len();

		for _ in 0..nb_epochs {

			println!("NEW EPOCH :");
			rng.shuffle(training_data);
			let mut mini_batches: Vec<&[(DVector<f64>, DVector<f64>)]> = Vec::new();

			let mut i = 0;
			while i + mini_batch_size < n {
				mini_batches.push(&training_data[i..(i + mini_batch_size)]);
				i += mini_batch_size;
			}

			println!("MINI BATCHES CREATED !");

			for mini_batch in mini_batches.iter() {
				self.update_mini_batch(mini_batch, learning_step);
			}
		}
	}
}