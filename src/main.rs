extern crate neural_net;
use neural_net::Network;
use neural_net::load_mnist::*;

fn main() {

	println!("Loading data...");
	let mut data = load_training_data().unwrap();
	println!("Data loaded");
	println!("Initializing network...");
	let mut net = Network::new(vec![IMAGE_SIZE, 30, 10]);
	println!("Network created");
	net.stochastic_gradient_descent(&mut data, 1, 10, 3.0);
}
