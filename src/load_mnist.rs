use std::io;
use std::io::prelude::*;
use std::fs::File;
extern crate nalgebra;
use nalgebra::DVector;

const NB_TRAINING_SAMPLES: usize = 50_000;
const NB_TEST_SAMPLES: usize = 10_000;
const SKIP_BYTES_TRAINING_IMAGES: usize = 16;
const SKIP_BYTES_TRAINING_LABELS: usize = 8;
pub const IMAGE_SIZE: usize = 784;
const LABEL_SIZE: usize = 1;
const TRAINING_IMAGES_PATH: &str  = "resources/train-images.idx3-ubyte";
const TRAINING_LABELS_PATH: &str  = "resources/train-labels.idx1-ubyte";


pub fn load_training_data() -> Result<Vec<(DVector<f64>, DVector<f64>)>, io::Error> {

	let mut res: Vec<(DVector<f64>, DVector<f64>)> = Vec::new();

	let mut train_images = File::open(TRAINING_IMAGES_PATH)?;
	let mut train_labels = File::open(TRAINING_LABELS_PATH)?;

	train_images.read(&mut [0; SKIP_BYTES_TRAINING_IMAGES])?;
	train_labels.read(&mut [0; SKIP_BYTES_TRAINING_LABELS])?;

	let mut image = [0; IMAGE_SIZE];
	let mut label = [0; LABEL_SIZE];
	
	for _ in 0..NB_TRAINING_SAMPLES {
		train_images.read(&mut image)?;
		train_labels.read(&mut label)?;
		let mut temp = DVector::from_element(10, 0.0);
		temp[label[0] as usize] = 1.0;
		res.push((DVector::from_row_slice(IMAGE_SIZE,
					&image.iter().map(|x| *x as f64).collect::<Vec<_>>()[..]),
				temp.clone()));
	}

	return Ok(res);
}