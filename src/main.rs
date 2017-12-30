use std::io;
use std::io::prelude::*;
use std::fs::File;


extern crate serde;
extern crate serde_json;
extern crate bincode;
extern crate rmp_serde;

extern crate nn;
use nn::Network;

extern crate nalgebra;
use nalgebra as na;
use na::DVector;

const NB_TRAIN_SAMPLES: usize = 50_000;
const NB_TEST_SAMPLES: usize = 10_000;
const SKIP_BYTES_TRAIN_IMAGES: usize = 16;
const SKIP_BYTES_TRAIN_LABELS: usize = 8;
const IMAGE_SIZE: usize = 784;
const LABEL_SIZE: usize = 1;


fn open_mnist_train() -> Vec<(DVector<f64>, DVector<f64>)> {

	let mut res: Vec<(DVector<f64>, DVector<f64>)> = Vec::new();

	let mut train_images = File::open("resources/train-images.idx3-ubyte").unwrap();
	let mut train_labels = File::open("resources/train-labels.idx1-ubyte").unwrap();

	train_images.read(&mut [0; SKIP_BYTES_TRAIN_IMAGES]).unwrap();
	train_labels.read(&mut [0; SKIP_BYTES_TRAIN_LABELS]).unwrap();

	let mut image = [0; IMAGE_SIZE];
	let mut label = [0; LABEL_SIZE];
	for i in 0..NB_TRAIN_SAMPLES {
		train_images.read(&mut image).unwrap();
		train_labels.read(&mut label).unwrap();
		res.push((DVector::from_row_slice(IMAGE_SIZE, &image.iter().map(|x| *x as f64).collect::<Vec<_>>()[..]),
					DVector::from_row_slice(LABEL_SIZE, &label.iter().map(|x| *x as f64).collect::<Vec<_>>()[..])));

		println!("{:?}", i);
	}

	return res;
}

fn main() {

	let data: Vec<(DVector<f64>, DVector<f64>)> = 
				bincode::deserialize_from(&mut File::open("resources/train_mnist.bincode").unwrap(),
					bincode::Infinite).unwrap();

}
