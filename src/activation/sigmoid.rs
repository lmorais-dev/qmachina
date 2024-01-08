use super::ActivationFunction;

/// `SigmoidActivationFunction` represents the sigmoid activation function
/// used in neural networks. The sigmoid function is a smooth, S-shaped function
/// that maps real-valued inputs to the (0, 1) range. It is widely used in scenarios
/// where a probability output is needed.
///
/// This structure implements the `ActivationFunction<f64, f64>` trait, meaning it
/// takes a `f64` as input and returns a `f64` as output. The sigmoid function is defined
/// as `1 / (1 + e^(-x))`, and its derivative, used in the backpropagation algorithm,
/// is `sigmoid(x) * (1 - sigmoid(x))`.
pub struct SigmoidActivationFunction;

impl ActivationFunction<f64, f64> for SigmoidActivationFunction {
    /// Computes the sigmoid of a given input value.
    ///
    /// The sigmoid function transforms the input into a value between 0 and 1,
    /// which can be interpreted as a probability or activation level.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which to compute the sigmoid.
    ///
    /// # Returns
    ///
    /// The sigmoid of the input, a value between 0 and 1.
    fn activate(&self, input: f64) -> f64 {
        1.0 / (1.0 + (-input).exp())
    }

    /// Computes the derivative of the sigmoid function for a given input value.
    ///
    /// The derivative of sigmoid is important for neural network training algorithms,
    /// specifically in the backpropagation step. It represents the rate of change of
    /// the sigmoid function at the given input.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which to compute the derivative.
    ///
    /// # Returns
    ///
    /// The derivative of the sigmoid function at the given input.
    fn derivate(&self, input: f64) -> f64 {
        let sigmoid = self.activate(input);
        sigmoid * (1.0 - sigmoid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_activate_positive() {
        let sigmoid = SigmoidActivationFunction;

        let output = sigmoid.activate(2.0);
        assert!(output > 0.5 && output < 1.0);
    }

    #[test]
    fn sigmoid_activate_negative() {
        let sigmoid = SigmoidActivationFunction;

        let output = sigmoid.activate(-2.0);
        assert!(output > 0.0 && output < 0.5);
    }

    #[test]
    fn sigmoid_activate_zero() {
        let sigmoid = SigmoidActivationFunction;

        let output = sigmoid.activate(0.0);
        assert_eq!(output, 0.5);
    }

    #[test]
    fn sigmoid_activate_positive_ex() {
        let sigmoid = SigmoidActivationFunction;

        let output = sigmoid.activate(1000.0);
        assert_eq!(output, 1.0);
    }

    #[test]
    fn sigmoid_activate_negative_ex() {
        let sigmoid = SigmoidActivationFunction;

        let output = sigmoid.activate(-1000.0);
        assert_eq!(output, 0.0);
    }

    #[test]
    fn sigmoid_derivate_positive() {
        let sigmoid = SigmoidActivationFunction;

        let output = sigmoid.derivate(1.0);
        assert!(output > 0.0 && output < 0.25);
    }

    #[test]
    fn sigmoid_derivate_negative() {
        let sigmoid = SigmoidActivationFunction;

        let output = sigmoid.derivate(-1.0);
        assert!(output > 0.0 && output < 0.25);
    }

    #[test]
    fn sigmoid_derivate_zero() {
        let sigmoid = SigmoidActivationFunction;

        let output = sigmoid.derivate(0.0);
        assert_eq!(output, 0.25);
    }

    #[test]
    fn sigmoid_derivate_positive_ex() {
        let sigmoid = SigmoidActivationFunction;

        let output = sigmoid.derivate(1000.0);
        assert_eq!(output, 0.0);
    }

    #[test]
    fn sigmoid_derivate_negative_ex() {
        let sigmoid = SigmoidActivationFunction;

        let output = sigmoid.derivate(1000.0);
        assert_eq!(output, 0.0);
    }
}