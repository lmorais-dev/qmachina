use super::ActivationFunction;

/// A `StepActivationFunction` represents a simple step activation function
/// used in neural networks. This function outputs 1.0 if the input is positive,
/// and 0.0 otherwise.
///
/// It is a basic type of activation function, often used in simple binary
/// classification problems. However, its usage is limited in complex networks
/// due to its non-differentiability at zero and the fact that its derivative is zero
/// almost everywhere.
///
/// Implements `ActivationFunction<f64, f64>`, accepting `f64` as input and returning
/// `f64` as output.
///
/// # Examples
///
/// ```
/// use quantify::activation::ActivationFunction;
/// use quantify::activation::step::StepActivationFunction;
/// 
/// let step_func = StepActivationFunction;
/// let activated_value = step_func.activate(0.5); // returns 1.0
/// let derivative_value = step_func.derivate(0.5); // returns 0.0
/// ```
pub struct StepActivationFunction;


impl ActivationFunction<f64, f64> for StepActivationFunction {
    /// Applies the step activation function to a given input.
    ///
    /// # Arguments
    ///
    /// * `input` - A `f64` value representing the input to the activation function.
    ///
    /// # Returns
    ///
    /// Returns 1.0 if the input is greater than 0.0, otherwise returns 0.0.
    fn activate(&self, input: f64) -> f64 {
        if input > 0.0 { 1.0 } else { 0.0 }
    }

    /// Returns the derivative of the step activation function.
    ///
    /// The derivative of a step function is 0 for all inputs, which is a
    /// simplification as the actual step function is not differentiable.
    ///
    /// # Arguments
    ///
    /// * `_`: A `f64` value representing the input. The input is not used
    ///   since the derivative is constant.
    ///
    /// # Returns
    ///
    /// Always returns 0.0.
    fn derivate(&self, _: f64) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_activation_positive() {
        let step_activation = StepActivationFunction;

        assert_eq!(step_activation.activate(1.0), 1.0);
        assert_eq!(step_activation.activate(5.0), 1.0);
        assert_eq!(step_activation.activate(0.1), 1.0);
    }

    #[test]
    fn step_activation_non_positive() {
        let step_activation = StepActivationFunction;

        assert_eq!(step_activation.activate(0.0), 0.0);
        assert_eq!(step_activation.activate(-1.0), 0.0);
        assert_eq!(step_activation.activate(-5.0), 0.0);
    }
}