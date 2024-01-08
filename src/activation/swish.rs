use super::{sigmoid::SigmoidActivationFunction, ActivationFunction};

/// `SwishActivationFunction` represents the Swish activation function
/// used in neural networks. Swish is defined as `x * sigmoid(beta * x)`.
/// It has been found to sometimes outperform traditional functions like ReLU
/// in deep neural networks.
///
/// This struct implements the `ActivationFunction<f64, f64>` trait. By default,
/// the beta parameter is set to 1.0, but it can be adjusted if needed.
///
/// # Examples
///
/// ```
/// use quantify::activation::ActivationFunction;
/// use quantify::activation::swish::SwishActivationFunction;
/// 
/// let swish = SwishActivationFunction::new(1.0);
/// let activated_value = swish.activate(0.5); // Example usage
/// ```
pub struct SwishActivationFunction {
    beta: f64,
    sigmoid: SigmoidActivationFunction,
}

impl SwishActivationFunction {
    /// Creates a new instance of `SwishActivationFunction` with the given beta value.
    ///
    /// # Arguments
    ///
    /// * `beta` - The beta value for the Swish function.
    ///
    /// # Returns
    ///
    /// A new instance of `SwishActivationFunction`.
    pub fn new(beta: f64) -> Self {
        SwishActivationFunction {
            beta,
            sigmoid: SigmoidActivationFunction,
        }
    }

    /// Updates the beta parameter of the Swish activation function.
    ///
    /// This method allows for dynamically adjusting the beta value, which can
    /// be useful in various training scenarios or experiments.
    ///
    /// # Arguments
    ///
    /// * `new_beta` - The new value to set for the beta parameter.
    pub fn update_beta(&mut self, new_beta: f64) {
        self.beta = new_beta;
    }
}

impl ActivationFunction<f64, f64> for SwishActivationFunction {
    /// Computes the Swish activation function for a given input value.
    ///
    /// The function is defined as `x * sigmoid(beta * x)`.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which to compute the Swish function.
    ///
    /// # Returns
    ///
    /// The Swish of the input.
    fn activate(&self, input: f64) -> f64 {
        input * self.sigmoid.activate(self.beta * input)
    }

    /// Computes the derivative of the Swish function for a given input value.
    ///
    /// The derivative involves both the sigmoid function and its derivative.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which to compute the derivative.
    ///
    /// # Returns
    ///
    /// The derivative of the Swish function at the given input.
    fn derivate(&self, input: f64) -> f64 {
        let sigmoid = self.sigmoid.activate(self.beta * input);
        sigmoid + self.beta * input * (1.0 - sigmoid)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swish_derivate_positive() {
        let swish = SwishActivationFunction::new(1.0);

        let input = 2.0;
        let output = swish.derivate(input);
        let sigmoid = swish.sigmoid.activate(input);
        let expected = sigmoid + input * (1.0 - sigmoid);
        assert!((output - expected).abs() < 1e-5);
    }

    #[test]
    fn swish_derivate_negative() {
        let swish = SwishActivationFunction::new(1.0);

        let input = -2.0;
        let output = swish.derivate(input);
        let sigmoid = swish.sigmoid.activate(input);
        let expected = sigmoid + input * (1.0 - sigmoid);
        assert!((output - expected).abs() < 1e-5);
    }

    #[test]
    fn swish_derivate_zero() {
        let swish = SwishActivationFunction::new(1.0);

        let input = 0.0;
        let output = swish.derivate(input);
        // At zero, the derivative of Swish should be equal to 0.5
        assert_eq!(output, 0.5);
    }

    #[test]
    fn swish_derivate_extreme_positive() {
        let swish = SwishActivationFunction::new(1.0);

        let input = 1000.0;
        let output = swish.derivate(input);
        // For large positive x, the derivative should approximate 1
        assert!((output - 1.0).abs() < 1e-3);
    }

    #[test]
    fn swish_derivate_extreme_negative() {
        let swish = SwishActivationFunction::new(1.0);

        let input = -1000.0;
        let output = swish.derivate(input);
        let sigmoid = swish.sigmoid.activate(input);
        let expected = sigmoid + input * (1.0 - sigmoid);
        // For large negative x, the derivative should be close to 0, but not exactly 0
        assert!((output - expected).abs() < 1e-3);
    }
}
