use super::ActivationFunction;

/// `PReLUActivationFunction` represents the Parametric Rectified Linear Unit (PReLU)
/// activation function used in neural networks. PReLU is a variant of the ReLU
/// function where the negative part of the function is not fixed but instead
/// parameterized by a learnable coefficient `alpha`.
///
/// This struct implements the `ActivationFunction<f64, f64>` trait. The PReLU function
/// is defined as `x` for `x > 0` and `alpha * x` for `x <= 0`. Unlike Leaky ReLU,
/// in PReLU, `alpha` is a parameter that can be learned during the training process.
///
/// # Examples
///
/// ```
/// use quantify::activation::ActivationFunction;
/// use quantify::activation::param_relu::PReLUActivationFunction;
/// 
/// let prelu = PReLUActivationFunction::new(0.25);
/// let activated_value = prelu.activate(-2.0); // returns -0.5
/// let derivative_value = prelu.derivate(-2.0); // returns 0.25
/// ```
pub struct PReLUActivationFunction {
    alpha: f64,
}

impl PReLUActivationFunction {
    /// Creates a new instance of `PReLUActivationFunction` with the given alpha value.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The initial value for the alpha coefficient, which will be learned
    ///             and adjusted during training.
    ///
    /// # Returns
    ///
    /// A new instance of `PReLUActivationFunction`.
    pub fn new(alpha: f64) -> Self {
        PReLUActivationFunction { alpha }
    }

    /// Updates the alpha parameter of the PReLU function.
    ///
    /// # Arguments
    ///
    /// * `new_alpha` - The new value to set for the alpha parameter.
    pub fn update_alpha(&mut self, new_alpha: f64) {
        self.alpha = new_alpha;
    }
}

impl ActivationFunction<f64, f64> for PReLUActivationFunction {
    /// Computes the Parametric Rectified Linear Unit (PReLU) of a given input value.
    ///
    /// For positive inputs, it returns the input value itself. For non-positive inputs,
    /// it returns `alpha * input`, where `alpha` is the learnable parameter.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which to compute the PReLU.
    ///
    /// # Returns
    ///
    /// The PReLU of the input.
    fn activate(&self, input: f64) -> f64 {
        if input > 0.0 {
            input
        } else {
            self.alpha * input
        }
    }

    /// Computes the derivative of the PReLU function for a given input value.
    ///
    /// For positive inputs, the derivative is 1. For non-positive inputs, the
    /// derivative is `alpha`.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which to compute the derivative.
    ///
    /// # Returns
    ///
    /// The derivative of the PReLU function at the given input.
    fn derivate(&self, input: f64) -> f64 {
        if input > 0.0 {
            1.0
        } else {
            self.alpha
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALPHA: f64 = 0.01;  // Assuming alpha is set to 0.01

    #[test]
    fn prelu_activate_positive() {
        let prelu = PReLUActivationFunction::new(ALPHA);

        let output = prelu.activate(2.0);
        assert_eq!(output, 2.0);
    }

    #[test]
    fn prelu_activate_negative() {
        let prelu = PReLUActivationFunction::new(ALPHA);

        let output = prelu.activate(-2.0);
        assert_eq!(output, -0.02);
    }

    #[test]
    fn prelu_activate_zero() {
        let prelu = PReLUActivationFunction::new(ALPHA);

        let output = prelu.activate(0.0);
        assert_eq!(output, 0.0);
    }

    #[test]
    fn prelu_activate_positive_ex() {
        let prelu = PReLUActivationFunction::new(ALPHA);

        let output = prelu.activate(1000.0);
        assert_eq!(output, 1000.0);
    }

    #[test]
    fn prelu_activate_negative_ex() {
        let prelu = PReLUActivationFunction::new(ALPHA);

        let output = prelu.activate(-1000.0);
        assert_eq!(output, -10.0);
    }

    #[test]
    fn prelu_derivate_positive() {
        let prelu = PReLUActivationFunction::new(ALPHA);

        let output = prelu.derivate(1.0);
        assert_eq!(output, 1.0);
    }

    #[test]
    fn prelu_derivate_negative() {
        let prelu = PReLUActivationFunction::new(ALPHA);

        let output = prelu.derivate(-1.0);
        assert_eq!(output, ALPHA);
    }

    #[test]
    fn prelu_derivate_zero() {
        let prelu = PReLUActivationFunction::new(ALPHA);

        let output = prelu.derivate(0.0);
        assert_eq!(output, ALPHA); // Note: The behavior at zero might depend on the implementation
    }

    #[test]
    fn prelu_derivate_positive_ex() {
        let prelu = PReLUActivationFunction::new(ALPHA);

        let output = prelu.derivate(1000.0);
        assert_eq!(output, 1.0);
    }

    #[test]
    fn prelu_derivate_negative_ex() {
        let prelu = PReLUActivationFunction::new(ALPHA);

        let output = prelu.derivate(-1000.0);
        assert_eq!(output, ALPHA);
    }
}
