extern crate rand;
use rand::{thread_rng, Rng};
use rand::distributions::{Normal, IndependentSample};

extern crate itertools;
use itertools::zip;

use std::io;
use std::fs::File;

extern crate serde;

 #[macro_use]
extern crate serde_derive;
extern crate serde_json;

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

#[derive(Serialize, Deserialize, Debug)]
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

		let mut rng = thread_rng();
		let normal = Normal::new(0.0, 1.0);

		for layer in 1..nb_layers {
			//biases.push(DVector::new_random(sizes[layer]));
			//weights.push(DMatrix::new_random(sizes[layer], sizes[layer - 1]));
			let mut bias: DVector<f64> = DVector::zeros(sizes[layer]);
			for x in &mut bias {
				*x = normal.ind_sample(&mut rng);
			}
			biases.push(bias);

			let mut weight: DMatrix<f64> = DMatrix::zeros(sizes[layer], sizes[layer - 1]);
			for x in &mut weight {
				*x = normal.ind_sample(&mut rng);
			}
			weights.push(weight)
		}

		return Network {sizes: sizes,
						weights: weights,
						biases: biases,
						nb_layers: nb_layers}
	}

	pub fn save(&self, path: &str) -> Result<(), io::Error> {

		let mut file = File::create(path)?;

		serde_json::to_writer(&mut file, &self)?;
		return Ok(());
	}

	pub fn load(path: &str) -> Result<Network, serde_json::Error> {
		let file = File::open(path).unwrap();
		return serde_json::from_reader(file);
	}

	pub fn feed_forward(&self, input: &DVector<f64>) -> DVector<f64> {

		let mut output: DVector<f64> = input.clone();
		for (w, b) in zip(&self.weights, &self.biases) {
			output = w*output + b;
			for x in &mut output {
				*x = sigmoid(*x);
			}
		}

		return output
	}

	pub fn evaluate(&self, test_data: &Vec<(DVector<f64>, DVector<f64>)>) {
		// MNIST SPECIFIC
		let mut correct = 0;
		let total = test_data.len();
		for &(ref input, ref expected_output) in test_data {
			let output = self.feed_forward(&input);
			let mut max = output[0];
			let mut imax = 0;
			for (i, x) in zip(0..output.len(), &output) {
				if *x > max {
					max = *x;
					imax = i;
				}
			}
			if expected_output[imax] == 1.0 {
				correct += 1;
			}

		}

		println!("{}/{}", correct, total);
	}

	fn backprop(&self, input: &DVector<f64>, expected_output: &DVector<f64>)
				-> (Vec<DVector<f64>>, Vec<DMatrix<f64>>) {



		let n = self.nb_layers - 1;

		let mut nabla_biases: Vec<Option<DVector<f64>>> = vec![None; n];
		let mut nabla_weights: Vec<Option<DMatrix<f64>>> = vec![None; n];

		let mut activation = input.clone();
		let mut activations: Vec<DVector<f64>> = Vec::with_capacity(self.nb_layers);
		activations.push(activation.clone());

		let mut z_layers: Vec<DVector<f64>> = Vec::with_capacity(n);
		for (w, b) in zip(&self.weights, &self.biases) {
			let z = w*&activation + b;
			z_layers.push(z.clone());
			activation = z.clone();
			for x in &mut activation {
				*x = sigmoid(*x);
			}
			activations.push(activation.clone());
		}

		let mut delta = cost_derivative(&activations[n], expected_output);

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

		for &(ref input, ref expected_output) in mini_batch {

			let (delta_nabla_b, delta_nabla_w) = self.backprop(input, expected_output);

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

		println!("Beginning stochastic gradient descent");
		let mut rng = thread_rng();
		let n = training_data.len();

		for i in 0..nb_epochs {

			println!("Epoch {}", i);
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
	}
}