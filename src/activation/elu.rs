use super::ActivationFunction;

/// `ELUActivationFunction` represents the Exponential Linear Unit (ELU) activation
/// function used in neural networks. ELU is designed to combine the advantages
/// of ReLU and its variants while offering a smoother and more continuous curve
/// that helps mitigate the dying neuron problem and reduce the vanishing gradient
/// effect.
///
/// This struct implements the `ActivationFunction<f64, f64>` trait. The ELU function
/// is defined as `x` for `x > 0` and `alpha * (e^x - 1)` for `x <= 0`, where `alpha`
/// is a hyperparameter.
///
/// # Examples
///
/// ```
/// use quantify::activation::ActivationFunction;
/// use quantify::activation::elu::ELUActivationFunction;
/// 
/// let elu = ELUActivationFunction::new(1.0);
/// let activated_value = elu.activate(-1.0);  // Evaluates to approximately -0.6321
/// let derivative_value = elu.derivate(-1.0); // Evaluates to approximately 0.3679
/// ```
pub struct ELUActivationFunction {
    alpha: f64,
}

impl ELUActivationFunction {
    /// Creates a new instance of `ELUActivationFunction` with the given alpha value.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The alpha value for the ELU function, controlling the value
    ///             to which an ELU saturates for negative net inputs.
    ///
    /// # Returns
    ///
    /// A new instance of `ELUActivationFunction`.
    pub fn new(alpha: f64) -> Self {
        ELUActivationFunction { alpha }
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

impl ActivationFunction<f64, f64> for ELUActivationFunction {
    /// Computes the Exponential Linear Unit (ELU) of a given input value.
    ///
    /// For positive inputs, it returns the input itself. For non-positive inputs,
    /// it returns `alpha * (e^x - 1)`, which provides a smooth curve that approaches
    /// `alpha` for large negative values.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which to compute the ELU.
    ///
    /// # Returns
    ///
    /// The ELU of the input.
    fn activate(&self, input: f64) -> f64 {
        if input > 0.0 {
            input
        } else {
            self.alpha * (input.exp() - 1.0)
        }
    }

    /// Computes the derivative of the ELU function for a given input value.
    ///
    /// For positive inputs, the derivative is 1. For non-positive inputs, the
    /// derivative is `alpha * e^x`, which smoothly approaches `alpha` as `x` decreases.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which to compute the derivative.
    ///
    /// # Returns
    ///
    /// The derivative of the ELU function at the given input.
    fn derivate(&self, input: f64) -> f64 {
        if input > 0.0 {
            1.0
        } else {
            self.alpha * input.exp()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALPHA: f64 = 0.01;  // Assuming alpha is set to 0.01

    #[test]
    fn elu_activate_positive() {
        let elu = ELUActivationFunction::new(ALPHA);

        let output = elu.activate(2.0);
        assert_eq!(output, 2.0);
    }

    #[test]
    fn elu_activate_negative() {
        let elu = ELUActivationFunction::new(ALPHA);

        let output = elu.activate(-2.0);
        assert!(output > -ALPHA && output < 0.0);
    }

    #[test]
    fn elu_activate_zero() {
        let elu = ELUActivationFunction::new(ALPHA);

        let output = elu.activate(0.0);
        assert_eq!(output, 0.0);
    }

    #[test]
    fn elu_activate_positive_ex() {
        let elu = ELUActivationFunction::new(ALPHA);

        let output = elu.activate(1000.0);
        assert_eq!(output, 1000.0);
    }

    #[test]
    fn elu_activate_negative_ex() {
        let elu = ELUActivationFunction::new(ALPHA);

        let output = elu.activate(-1000.0);
        assert!(output > -1.0); // ELU saturates near -ALPHA for large negative inputs
    }

    #[test]
    fn elu_derivate_positive() {
        let elu = ELUActivationFunction::new(ALPHA);

        let output = elu.derivate(1.0);
        assert_eq!(output, 1.0);
    }

    #[test]
    fn elu_derivate_negative() {
        let elu = ELUActivationFunction::new(ALPHA);

        let output = elu.derivate(-1.0);
        let expected = ALPHA * (-1.0f64).exp();
        assert_eq!(output, expected);
    }

    #[test]
    fn elu_derivate_zero() {
        let elu = ELUActivationFunction::new(ALPHA);

        let output = elu.derivate(0.0);
        assert_eq!(output, ALPHA);
    }

    #[test]
    fn elu_derivate_positive_ex() {
        let elu = ELUActivationFunction::new(ALPHA);

        let output = elu.derivate(1000.0);
        assert_eq!(output, 1.0);
    }

    #[test]
    fn elu_derivate_negative_ex() {
        let elu = ELUActivationFunction::new(ALPHA);

        let output = elu.derivate(-1000.0);
        assert_eq!(output, 0.0);
    }
}
