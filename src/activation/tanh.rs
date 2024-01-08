use super::ActivationFunction;

/// `TanhActivationFunction` represents the hyperbolic tangent (tanh) activation function
/// used in neural networks. The tanh function is an S-shaped curve that maps real-valued
/// inputs to a range between -1 and 1. It is similar to the sigmoid function but with
/// output values centered around zero, making it useful in scenarios where negative
/// outputs are meaningful.
///
/// This struct implements the `ActivationFunction<f64, f64>` trait, taking a `f64` as input
/// and producing a `f64` as output. The tanh function is defined as `(e^(x) - e^(-x)) / (e^(x) + e^(-x))`,
/// and its derivative is `1 - tanh(x)^2`, both of which are important in neural network training.
pub struct TanhActivationFunction;

impl ActivationFunction<f64, f64> for TanhActivationFunction {
    /// Computes the hyperbolic tangent of a given input value.
    ///
    /// The tanh function scales the input to be within the range of -1 to 1,
    /// providing a smoothly varying value that is centered around zero.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which to compute the hyperbolic tangent.
    ///
    /// # Returns
    ///
    /// The hyperbolic tangent of the input.
    fn activate(&self, input: f64) -> f64 {
        input.tanh()
    }

    /// Computes the derivative of the hyperbolic tangent function for a given input value.
    ///
    /// The derivative of tanh is important in the context of neural network training,
    /// especially for backpropagation. It represents the rate of change of the tanh
    /// function at a given input value.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which to compute the derivative.
    ///
    /// # Returns
    ///
    /// The derivative of the hyperbolic tangent function at the given input.
    fn derivate(&self, input: f64) -> f64 {
        1.0 - input.tanh().powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tanh_activate_positive() {
        let tanh = TanhActivationFunction;

        let output = tanh.activate(2.0);
        assert!(output > 0.0 && output < 1.0);
    }

    #[test]
    fn tanh_activate_negative() {
        let tanh = TanhActivationFunction;

        let output = tanh.activate(-2.0);
        assert!(output < 0.0 && output > -1.0);
    }

    #[test]
    fn tanh_activate_zero() {
        let tanh = TanhActivationFunction;

        let output = tanh.activate(0.0);
        assert_eq!(output, 0.0);
    }

    #[test]
    fn tanh_activate_extreme_positive() {
        let tanh = TanhActivationFunction;

        let output = tanh.activate(1000.0);
        assert_eq!(output, 1.0);
    }

    #[test]
    fn tanh_activate_extreme_negative() {
        let tanh = TanhActivationFunction;

        let output = tanh.activate(-1000.0);
        assert_eq!(output, -1.0);
    }

    #[test]
    fn tanh_derivate_positive() {
        let tanh = TanhActivationFunction;

        let input = 1.0;
        let output = tanh.derivate(input);
        let expected = 1.0 - tanh.activate(input).powi(2);
        assert_eq!(output, expected);
    }

    #[test]
    fn tanh_derivate_negative() {
        let tanh = TanhActivationFunction;

        let input = -1.0;
        let output = tanh.derivate(input);
        let expected = 1.0 - tanh.activate(input).powi(2);
        assert_eq!(output, expected);
    }

    #[test]
    fn tanh_derivate_zero() {
        let tanh = TanhActivationFunction;

        let output = tanh.derivate(0.0);
        assert_eq!(output, 1.0);
    }
}
