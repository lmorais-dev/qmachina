use super::ActivationFunction;

/// Represents the sigmoid activation function in neural networks.
///
/// The sigmoid function is a smooth, S-shaped curve that maps real-valued 
/// inputs to a range between 0 and 1. It is commonly used in binary 
/// classification problems and other scenarios where a probability-like 
/// output is needed.
///
/// This structure implements the `ActivationFunction<f64, f64>` trait. It
/// takes a `f64` input and returns a `f64` output. The sigmoid function is 
/// defined mathematically as `1 / (1 + e^(-x))`. Its derivative, which is 
/// `sigmoid(x) * (1 - sigmoid(x))`, plays a critical role in neural network 
/// training algorithms, particularly in the backpropagation process.
///
/// # Mathematical Background
///
/// The sigmoid function smoothly transitions inputs to outputs in the range (0, 1),
/// making it suitable for scenarios like logistic regression and neurons in a
/// binary classification neural network. Its "S" shape allows it to handle inputs 
/// of varied scales effectively.
///
/// # Usage
///
/// Sigmoid can be used as an activation function in a neural network layer to
/// normalize outputs. It's particularly useful in the final layer of a binary
/// classification network, where the output can be interpreted as a probability.
///
/// # Example
///
/// ```
/// use qmachina::activation::ActivationFunction;
/// use qmachina::activation::sigmoid::SigmoidActivationFunction;
/// 
/// let sigmoid = SigmoidActivationFunction;
/// let input = 0.5;
/// let output = sigmoid.activate(input);
/// // `output` is now the sigmoid of 0.5, a value between 0 and 1.
///
/// let derivative = sigmoid.derivate(input);
/// // `derivative` represents the rate of change of the sigmoid function at 0.5.
/// ```
///
/// Note: In practice, the derivative of the sigmoid function is often used in 
/// conjunction with the original function output, optimizing computational efficiency.
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