use ndarray::Array1;
use rand::Rng;

pub struct Node {
    weights: Array1<f64>,
    bias: f64,
    learning_rate: f64,
}

impl Node {
    pub fn new(input_size: usize, learning_rate: f64) -> Node {
        let mut rng = rand::thread_rng();
        let weights = Array1::from_shape_fn(input_size, |_| rng.gen_range(-1.0..1.0));
        let bias = rng.gen_range(-0.01..0.01);

        Node {
            weights,
            bias,
            learning_rate,
        }
    }

    pub fn predict(&self, input: Array1<f64>) -> f64 {
        let weighted_sum = self.weights.dot(&input) + self.bias;
        weighted_sum
    }
}

// Test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict() {
        use ndarray::Array1;
        let mut node = Node::new(3, 0.01);
        // Imposta pesi e bias noti per il test
        node.weights = Array1::from_vec(vec![0.5, -0.2, 0.3]);
        node.bias = 0.1;

        let inputs = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let output = node.predict(inputs);

        // Calcolo manuale dell'output atteso: (0.5*1.0) + (-0.2*2.0) + (0.3*3.0) + 0.1
        let expected_output = (0.5 * 1.0) + (-0.2 * 2.0) + (0.3 * 3.0) + 0.1;
        assert!((output - expected_output).abs() < 1e-6, "Expected {} but got {}", expected_output, output);
    }
}
