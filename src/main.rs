use colored::{ColoredString, Colorize};
use csv::{Reader, StringRecord};
use image::{imageops::FilterType, io::Reader as ImageReader};
use ndarray::{array, rcarr1, Array, Array1, Array2, Axis};
use ndarray_rand::{
    rand_distr::{num_traits::Pow, Uniform},
    RandomExt,
};
use ndarray_stats::QuantileExt;
use std::{fs::File, io, ops::Add, time::Instant};

// w = weights, b = bias, i = input, h = hidden, o = output, l = label
// e.g. w_i_h = weights from input layer to hidden layer
const INPUT_LAYER_SIZE: usize = 784;
const HIDDEN_LAYER_SIZE: usize = 20;
const OUTPUT_LAYER_SIZE: usize = 10;
const DIGIT_COUNT: usize = 10; // 0,1,..,9
const TRAINING_SET_SIZE: usize = 60_000;
const TESTING_SET_SIZE: usize = 10_000;
const IMG_SITE_PX: usize = 28;
const TRAINING_SET_PATH: &str = "data/mnist_train.csv";
const TESTING_SET_PATH: &str = "data/mnist_test.csv";

fn main() {
    let learn_rate = 0.01;
    let mut nr_correct = 0;
    let epochs = 3;

    let distribution = Uniform::new(-0.5, 0.5);
    let mut w_i_h = Array::random((HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE), distribution);
    let mut w_h_o = Array::random((OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE), distribution);
    let mut b_i_h = Array1::<f64>::zeros(HIDDEN_LAYER_SIZE);
    let mut b_h_o = Array1::<f64>::zeros(OUTPUT_LAYER_SIZE);

    // Network training
    for epoch in 0..epochs {
        println!("{}", format!("{:=^28}", format!("[ Epoch {} ]", &epoch)));
        let time = Instant::now();

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path("data/mnist_train.csv")
            .unwrap();

        let mut image_count = 0;
        for result in reader.records() {
            let record = result.unwrap();
            let (label, image) = read_data(record);

            // Forward propagation -> hidden layer
            let h_pre = &b_i_h + &w_i_h.dot(&image);
            let h = h_pre.mapv(|x| 1.0 / (1.0 + (-x).exp()));

            // Forward propagation -> output layer
            let o_pre = &b_h_o + &w_h_o.dot(&h);
            let o = o_pre.mapv(|x| 1.0 / (1.0 + (-x).exp()));

            // Cost / Error calculation
            let l = to_one_hot(label);
            let e = calculate_error(&o, &l);
            nr_correct += (o.argmax().unwrap() == label) as i32;

            // Backward propagation output -> hidden (cost function derivative)
            let delta_o = o - l;
            w_h_o += &(-learn_rate
                * (&delta_o.clone().into_shape((10, 1)).unwrap())
                    .dot(&h.clone().into_shape((1, 20)).unwrap()));
            b_h_o += &(-learn_rate * &delta_o);

            // Backward propagation hidden -> input
            let delta_h = &delta_o.dot(&w_h_o) * &h * (-&h).add(1.0);
            w_i_h += &(-learn_rate
                * (&delta_h.clone().into_shape((HIDDEN_LAYER_SIZE, 1)).unwrap())
                    .dot(&image.into_shape((1, INPUT_LAYER_SIZE)).unwrap()));
            b_i_h += &(-learn_rate * delta_h);

            image_count += 1;
        }

        // Accuracy for each epoch
        println!(
            "Accuracy ... {:.2} %",
            (nr_correct as f64 / image_count as f64) * 100.0
        );
        println!(
            "Training time ... {}.{} s",
            time.elapsed().as_secs(),
            time.elapsed().subsec_millis()
        );
        nr_correct = 0;
    }

    // Digit recognition
    loop {
        let data = get_data_to_recognize();
        match data {
            Some((label, image)) => {
                // Print image
                let frame = "-".repeat(IMG_SITE_PX);
                println!("{}", frame);
                match label {
                    Some(l) => println!("{}", format!("{:-^28}", format!("( {} )", l))),
                    _ => (),
                }
                print_image(
                    rcarr1(&image.to_vec())
                        .reshape((IMG_SITE_PX, IMG_SITE_PX))
                        .to_owned(),
                );
                println!("{}", frame);

                let time = Instant::now();

                // Forward propagation -> hidden layer
                let h_pre = &b_i_h + &w_i_h.dot(&image);
                let h = h_pre.mapv(|x| 1.0 / (1.0 + (-x).exp()));

                // Forward propagation -> output layer
                let o_pre = &b_h_o + &w_h_o.dot(&h);
                let o = o_pre.mapv(|x| 1.0 / (1.0 + (-x).exp()));

                // Printing results
                let recognition_time = time.elapsed();
                let argmax = o.argmax().unwrap();
                o.iter().enumerate().for_each(|(i, val)| {
                    let mut fmt_val = ColoredString::from(format!("{:.2}", val * 100.0).as_str());
                    match label {
                        Some(l) if argmax == l && i == argmax => fmt_val = fmt_val.green(),
                        Some(_) if i == argmax => fmt_val = fmt_val.red(),
                        None if i == argmax => fmt_val = fmt_val.bold(),
                        _ => (),
                    }
                    print!("{}\t ... {: >6} %", i, fmt_val);
                    if i == argmax {
                        print!(" <-- MAX");
                    }
                    match label {
                        Some(l) if l == i => print!(" <-- TRUE"),
                        _ => (),
                    }
                    print!("\n");
                });
                println!("\nRecognition time ... {} μs", recognition_time.as_micros());
            }
            None => continue,
        }
    }
}

fn get_data_to_recognize() -> Option<(Option<usize>, Array1<f64>)> {
    println!(
        "\nSelect source of image to recognize:\n\tTraining set\t... 1\n\tTesting set\t... 2\n\tImage file\t... 3");
    let mut line = String::new();
    match io::stdin().read_line(&mut line) {
        Ok(_) => match line.trim_end() {
            "1" => get_img_from_train_set(),
            "2" => get_img_from_test_set(),
            "3" => get_img_from_file(),
            _ => None,
        },
        Err(_) => None,
    }
}

fn get_img_from_train_set() -> Option<(Option<usize>, Array1<f64>)> {
    println!(
        "\nSelect image from training set to recognize via its index from interval <0,{}):",
        TRAINING_SET_SIZE
    );
    match get_index() {
        Some(index) if index < TRAINING_SET_SIZE => {
            let mut reader = get_reader(TRAINING_SET_PATH);
            let (label, image) = get_data_at(&mut reader, index);
            Some((Some(label), image))
        }
        Some(i) => {
            println!("Index {} is out of range.", i);
            return None;
        }
        _ => None,
    }
}

fn get_img_from_test_set() -> Option<(Option<usize>, Array1<f64>)> {
    println!(
        "\nSelect image from testing to recognize via its index from interval <0,{}):",
        TESTING_SET_SIZE
    );
    match get_index() {
        Some(index) if index < TESTING_SET_SIZE => {
            let mut reader = get_reader(TESTING_SET_PATH);
            let (label, image) = get_data_at(&mut reader, index);
            Some((Some(label), image))
        }
        Some(i) => {
            println!("Index {} is out of range.", i);
            return None;
        }
        _ => None,
    }
}

fn get_img_from_file() -> Option<(Option<usize>, Array1<f64>)> {
    println!(
        "\nPath to approx. square image with centered black digit on white background (image will be resized to resolution {}x{} px): ",
        IMG_SITE_PX, IMG_SITE_PX
    );
    let mut line = String::new();
    match io::stdin().read_line(&mut line) {
        Ok(_) => match ImageReader::open(&line.trim_end()) {
            Ok(img) => match img.decode() {
                Ok(dec_img) => Some((
                    None,
                    Array1::from_vec(
                        dec_img
                            .resize(
                                IMG_SITE_PX as u32,
                                IMG_SITE_PX as u32,
                                FilterType::CatmullRom,
                            )
                            .into_luma8()
                            .iter()
                            .map(|p| ((255 - *p) as f64) / 255.0)
                            .collect::<Vec<f64>>(),
                    ),
                )),
                _ => None,
            },
            _ => None,
        },
        Err(_) => None,
    }
}

fn get_index() -> Option<usize> {
    let mut line = String::new();
    match io::stdin().read_line(&mut line) {
        Ok(_) => match line.trim_end().parse() {
            Ok(number) => Some(number),
            _ => None,
        },
        Err(_) => None,
    }
}

fn get_data_at(reader: &mut Reader<File>, index: usize) -> (usize, Array1<f64>) {
    let record = reader.records().into_iter().nth(index).unwrap().unwrap();

    read_data(record)
}

fn read_data(record: StringRecord) -> (usize, Array1<f64>) {
    let vector: Vec<f64> = record.iter().map(|s| s.parse().unwrap()).collect();
    let label = vector[0] as usize;
    let image = Array1::from_vec(vector[1..].to_vec()) / 255.0;

    (label, image)
}

fn get_reader(path: &str) -> Reader<File> {
    csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap()
}

// e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
fn calculate_error(output: &Array1<f64>, label: &Array1<f64>) -> f64 {
    let axis = Axis(0);
    let len = output.len_of(axis);
    let error: f64 = output
        .iter()
        .zip(label)
        .map(|(o, l)| (o - l).pow(2) / (len as f64))
        .sum();

    error
}

fn to_one_hot(label: usize) -> Array1<f64> {
    let mut one_hot = Array1::<f64>::zeros(DIGIT_COUNT);
    one_hot[label] = 1.0;

    one_hot
}

fn print_image(image: Array2<f64>) {
    let mut text: String = String::new();
    for row in image.axis_iter(Axis(0)) {
        for value in row.iter() {
            let ch = match value {
                v if *v < 0.2 => ' ',
                v if *v < 0.4 => '░',
                v if *v < 0.6 => '▒',
                v if *v < 0.8 => '▓',
                _ => '█',
            };
            text.push(ch);
        }
        text.push('\n');
    }
    println!("{}", text);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_image_1() {
        let actual = Array1::from_vec(
            ImageReader::open("data/test_1.png")
                .unwrap()
                .decode()
                .unwrap()
                .resize(
                    IMG_SITE_PX as u32,
                    IMG_SITE_PX as u32,
                    FilterType::CatmullRom,
                )
                .into_luma8()
                .iter()
                .map(|p| ((255 - *p) as f64) / 255.0)
                .collect::<Vec<f64>>(),
        );
        let mut expected = Array1::zeros(IMG_SITE_PX * IMG_SITE_PX);
        expected[1] = 1.0f64;
        assert_eq!(actual, expected);
    }
}
