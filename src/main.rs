extern crate neural_net;
use neural_net::Network;
use neural_net::load_mnist::*;

extern crate nalgebra;
use nalgebra::DVector;

fn main() {

	println!("Loading data...");
	let mut data = load_training_data().unwrap();
	println!("Data loaded");
	println!("Initializing network...");
	let mut net = Network::new(vec![IMAGE_SIZE, 30, 10]);
	println!("Network created");
	println!("{:?}", net.biases[0]);
	net.stochastic_gradient_descent(&mut data, 30, 10, 3.0);
	println!("training done");
	net.evaluate(&data);
	//println!("{:?}", net.feed_forward(&DVector::from_element(IMAGE_SIZE, 0.0)));
	//println!("{:?}", net.weights[0]);
	println!("biases:");
	println!("{:?}", net.biases[0]);
}
