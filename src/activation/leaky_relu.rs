use super::ActivationFunction;

/// `LeakyReLUActivationFunction` represents the Leaky Rectified Linear Unit (Leaky ReLU)
/// activation function used in neural networks. Unlike the standard ReLU, Leaky ReLU allows
/// a small, non-zero gradient when the unit is not active, thus addressing the problem of
/// dying neurons. It's particularly useful in deep learning models where this issue is prevalent.
///
/// This struct implements the `ActivationFunction<f64, f64>` trait. The Leaky ReLU function
/// is defined as `x` if `x > 0`, and `alpha * x` otherwise, where `alpha` is a small constant.
/// Its derivative is 1 for positive inputs and `alpha` for non-positive inputs.
///
/// # Examples
///
/// Implementing a LeakyReLU with a small negative slope (alpha):
///
/// ```
/// use qmachina::activation::ActivationFunction;
/// use qmachina::activation::leaky_relu::LeakyReLUActivationFunction;
/// 
/// let leaky_relu = LeakyReLUActivationFunction;
/// let activated_value = leaky_relu.activate(-5.0); // returns -0.05
/// let derivative_value = leaky_relu.derivate(-5.0); // returns 0.01
/// ```
pub struct LeakyReLUActivationFunction;

impl ActivationFunction<f64, f64> for LeakyReLUActivationFunction {
    /// Computes the Leaky Rectified Linear Unit (Leaky ReLU) of a given input value.
    ///
    /// If the input is positive, it returns the input value itself. For non-positive inputs,
    /// it returns `alpha` times the input value, allowing a small gradient when inactive.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which to compute the Leaky ReLU.
    ///
    /// # Returns
    ///
    /// The Leaky ReLU of the input.
    fn activate(&self, input: f64) -> f64 {
        if input > 0.0 { input } else { 0.01 * input }
    }

    /// Computes the derivative of the Leaky ReLU function for a given input value.
    ///
    /// The derivative is 1 for all positive input values and `alpha` for all non-positive values.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which to compute the derivative.
    ///
    /// # Returns
    ///
    /// The derivative of the Leaky ReLU function at the given input.
    fn derivate(&self, input: f64) -> f64 {
        if input > 0.0 { 1.0 } else { 0.01 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaky_relu_activate_positive() {
        let leaky_relu = LeakyReLUActivationFunction;

        let output = leaky_relu.activate(2.0);
        assert_eq!(output, 2.0);
    }

    #[test]
    fn leaky_relu_activate_negative() {
        let leaky_relu = LeakyReLUActivationFunction;

        let output = leaky_relu.activate(-2.0);
        assert_eq!(output, -0.02);
    }

    #[test]
    fn leaky_relu_activate_zero() {
        let leaky_relu = LeakyReLUActivationFunction;

        let output = leaky_relu.activate(0.0);
        assert_eq!(output, 0.0);
    }

    #[test]
    fn leaky_relu_activate_positive_ex() {
        let leaky_relu = LeakyReLUActivationFunction;

        let output = leaky_relu.activate(1000.0);
        assert_eq!(output, 1000.0);
    }

    #[test]
    fn leaky_relu_activate_negative_ex() {
        let leaky_relu = LeakyReLUActivationFunction;

        let output = leaky_relu.activate(-1000.0);
        assert_eq!(output, -10.0);
    }

    #[test]
    fn leaky_relu_derivate_positive() {
        let leaky_relu = LeakyReLUActivationFunction;

        let output = leaky_relu.derivate(1.0);
        assert_eq!(output, 1.0);
    }

    #[test]
    fn leaky_relu_derivate_negative() {
        let leaky_relu = LeakyReLUActivationFunction;

        let output = leaky_relu.derivate(-1.0);
        assert_eq!(output, 0.01);
    }

    #[test]
    fn leaky_relu_derivate_zero() {
        let leaky_relu = LeakyReLUActivationFunction;

        let output = leaky_relu.derivate(0.0);
        assert_eq!(output, 0.01); // Note: The behavior at zero might depend on the implementation
    }

    #[test]
    fn leaky_relu_derivate_positive_ex() {
        let leaky_relu = LeakyReLUActivationFunction;

        let output = leaky_relu.derivate(1000.0);
        assert_eq!(output, 1.0);
    }

    #[test]
    fn leaky_relu_derivate_negative_ex() {
        let leaky_relu = LeakyReLUActivationFunction;

        let output = leaky_relu.derivate(-1000.0);
        assert_eq!(output, 0.01);
    }
}
