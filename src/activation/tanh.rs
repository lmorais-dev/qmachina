use super::ActivationFunction;

/// Represents the hyperbolic tangent (tanh) activation function in neural networks.
///
/// The hyperbolic tangent function is an S-shaped curve mapping real-valued inputs
/// to outputs in the range of -1 to 1. It is akin to the sigmoid function but differs
/// in that its output values are centered around zero. This characteristic makes it 
/// particularly useful in scenarios where negative outputs are meaningful, such as 
/// when handling symmetrical data distributions.
///
/// This struct implements the `ActivationFunction<f64, f64>` trait. It accepts a 
/// `f64` input and produces a `f64` output. The tanh function is mathematically 
/// defined as `(e^x - e^(-x)) / (e^x + e^(-x))`. Its derivative, crucial in neural 
/// network training algorithms like backpropagation, is `1 - tanh(x)^2`.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use qmachina::activation::ActivationFunction;
/// use qmachina::activation::tanh::TanhActivationFunction;
/// 
/// let tanh_activation = TanhActivationFunction;
/// let input = 0.5;
/// let output = tanh_activation.activate(input);
/// ```
///
/// When used in a neural network layer, the tanh function can be applied to each neuron's output:
///
/// ```
/// use qmachina::activation::ActivationFunction;
/// use qmachina::activation::tanh::TanhActivationFunction;
/// 
/// fn apply_activation(neurons: Vec<f64>) -> Vec<f64> {
///     neurons.into_iter().map(|n| TanhActivationFunction.activate(n)).collect()
/// }
///
/// let layer_output = vec![0.5, -0.5, 0.0];
/// let activated_output = apply_activation(layer_output);
/// ```
///
/// Computing the derivative, useful in backpropagation:
///
/// ```
/// use qmachina::activation::ActivationFunction;
/// use qmachina::activation::tanh::TanhActivationFunction;
/// 
/// let tanh_activation = TanhActivationFunction;
/// let input = 0.5;
/// let derivative = tanh_activation.derivate(input);
/// ```
pub struct TanhActivationFunction;

impl ActivationFunction<f64, f64> for TanhActivationFunction {
    /// Computes the hyperbolic tangent of a given input value.
    ///
    /// The function transforms the input to a smoothly varying output within the
    /// range of -1 to 1, with a natural zero-centered property. This transformation
    /// aids in normalizing the input data and mitigating issues with gradient-based
    /// optimization methods in neural networks.
    ///
    /// # Arguments
    ///
    /// * `input` - A `f64` input value for which the hyperbolic tangent is computed.
    ///
    /// # Returns
    ///
    /// The `f64` hyperbolic tangent of the input.
    fn activate(&self, input: f64) -> f64 {
        input.tanh()
    }

    /// Computes the derivative of the hyperbolic tangent function at a given input value.
    ///
    /// This derivative is a critical component in neural network training processes,
    /// particularly during the backpropagation phase. It indicates how much the
    /// tanh function output changes concerning a change in its input.
    ///
    /// # Arguments
    ///
    /// * `input` - A `f64` input value for which the derivative of tanh is computed.
    ///
    /// # Returns
    ///
    /// The `f64` derivative of the hyperbolic tangent function at the specified input.
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
