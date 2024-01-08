use super::ActivationFunction;

/// Represents the Rectified Linear Unit (ReLU) activation function in neural networks.
///
/// ReLU is widely used in deep learning models. It outputs the input value if positive,
/// and zero otherwise. Known for its simplicity and efficiency, ReLU helps mitigate
/// the vanishing gradient problem common in deep networks and speeds up training.
///
/// This struct implements the `ActivationFunction<f64, f64>` trait. It accepts a `f64`
/// input and returns a `f64` output. The ReLU function is defined as `max(0, x)`.
/// Its derivative is 1 for positive inputs and 0 for non-positive inputs, which is
/// crucial during the backpropagation process in neural network training.
///
/// # Mathematical Background
///
/// The ReLU function is defined mathematically as:
/// `ReLU(x) = max(0, x)`.
/// It is linear for all positive values and zero for negative values, making it
/// computationally efficient and less susceptible to problems during optimization.
///
/// # Usage
///
/// ReLU is commonly used in hidden layers of neural networks. Its linear nature for
/// positive values allows models to learn complex patterns efficiently.
///
/// # Example
///
/// ```
/// use qmachina::activation::ActivationFunction;
/// use qmachina::activation::relu::ReLUActivationFunction;
/// 
/// let relu = ReLUActivationFunction;
/// let input = 0.5;
/// let output = relu.activate(input);
/// // `output` is now the ReLU of 0.5, which is 0.5.
///
/// let negative_input = -0.5;
/// let negative_output = relu.activate(negative_input);
/// // `negative_output` is the ReLU of -0.5, which is 0.
///
/// let derivative_positive = relu.derivate(input);
/// // `derivative_positive` is 1, as the input is positive.
///
/// let derivative_negative = relu.derivate(negative_input);
/// // `derivative_negative` is 0, as the input is negative.
/// ```
///
/// Note: While the ReLU function's derivative at zero is technically undefined,
/// it is conventionally treated as 0 for simplicity in most implementations.
pub struct ReLUActivationFunction;

impl ActivationFunction<f64, f64> for ReLUActivationFunction {
    /// Computes the Rectified Linear Unit (ReLU) of a given input value.
    ///
    /// The ReLU function returns the input value if it's positive, and 0 if it's negative
    /// or zero, effectively 'clipping' negative values to zero.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which to compute the ReLU.
    ///
    /// # Returns
    ///
    /// The ReLU of the input, which is either the input itself (if positive) or 0.
    fn activate(&self, input: f64) -> f64 {
        if input > 0.0 { input } else { 0.0 }
    }

    /// Computes the derivative of the ReLU function for a given input value.
    ///
    /// In the case of ReLU, the derivative is straightforward: it is 1 for all positive
    /// input values and 0 for all non-positive values. This is used in backpropagation
    /// to compute gradients.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which to compute the derivative.
    ///
    /// # Returns
    ///
    /// The derivative of the ReLU function at the given input, which is 1 if the input
    /// is positive and 0 otherwise.
    fn derivate(&self, input: f64) -> f64 {
        if input > 0.0 { 1.0 } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_activate_positive() {
        let relu = ReLUActivationFunction;

        let output = relu.activate(2.0);
        assert_eq!(output, 2.0);
    }

    #[test]
    fn relu_activate_negative() {
        let relu = ReLUActivationFunction;

        let output = relu.activate(-2.0);
        assert_eq!(output, 0.0);
    }

    #[test]
    fn relu_activate_zero() {
        let relu = ReLUActivationFunction;

        let output = relu.activate(0.0);
        assert_eq!(output, 0.0);
    }

    #[test]
    fn relu_activate_positive_ex() {
        let relu = ReLUActivationFunction;

        let output = relu.activate(1000.0);
        assert_eq!(output, 1000.0);
    }

    #[test]
    fn relu_activate_negative_ex() {
        let relu = ReLUActivationFunction;

        let output = relu.activate(-1000.0);
        assert_eq!(output, 0.0);
    }

    #[test]
    fn relu_derivate_positive() {
        let relu = ReLUActivationFunction;

        let output = relu.derivate(1.0);
        assert_eq!(output, 1.0);
    }

    #[test]
    fn relu_derivate_negative() {
        let relu = ReLUActivationFunction;

        let output = relu.derivate(-1.0);
        assert_eq!(output, 0.0);
    }

    #[test]
    fn relu_derivate_zero() {
        let relu = ReLUActivationFunction;

        let output = relu.derivate(0.0);
        assert_eq!(output, 0.0); // Note: This is a point of ambiguity in ReLU's derivative
    }

    #[test]
    fn relu_derivate_positive_ex() {
        let relu = ReLUActivationFunction;

        let output = relu.derivate(1000.0);
        assert_eq!(output, 1.0);
    }

    #[test]
    fn relu_derivate_negative_ex() {
        let relu = ReLUActivationFunction;

        let output = relu.derivate(-1000.0);
        assert_eq!(output, 0.0);
    }
}
