extern crate neural_net;
use neural_net::Network;
use neural_net::load_mnist::*;

fn main() {

	println!("Loading data...");
	let (mut train_data, test_data) = load_data().unwrap();
	println!("Data loaded");
	println!("Initializing network...");
	let mut net = Network::new(vec![IMAGE_SIZE, 30, 10]);
	println!("Network created");
	net.stochastic_gradient_descent(&mut train_data, 30, 10, 0.5);
	println!("training done");
	net.evaluate(&test_data);
	println!("saving...");
	net.save("out.json").unwrap();
	println!("saved, now loading");
	let new_net = Network::load("out.json").unwrap();
	new_net.evaluate(&test_data);

}
