use super::ActivationFunction;

/// Represents the Softmax activation function in neural networks.
///
/// Softmax is commonly used in the output layer of neural networks for 
/// multi-class classification problems. It transforms a vector of real 
/// values into a probability distribution. Each value's probability in the 
/// output vector is proportional to the exponential of the value, ensuring 
/// that all probabilities sum up to one and each is in the range (0, 1).
///
/// This struct implements the `ActivationFunction<Vec<f64>, Vec<f64>>` trait,
/// allowing it to be seamlessly integrated into neural network architectures
/// that handle multi-dimensional data.
///
/// # Mathematical Background
///
/// Given an input vector `x`, the Softmax function computes the exponential
/// of each element, subtracts the maximum element for numerical stability, and
/// then normalizes these values by dividing by the sum of all these exponentials.
///
/// # Usage
///
/// When applied to the raw output (logits) of a neural network layer, Softmax
/// converts these logits into probabilities, which are essential for tasks like
/// classification where probabilities are required for each class.
///
/// # Example
///
/// ```
/// use qmachina::activation::ActivationFunction;
/// use qmachina::activation::softmax::SoftmaxActivationFunction;
/// 
/// let softmax = SoftmaxActivationFunction;
/// let logits = vec![1.0, 2.0, 3.0];
/// let probabilities = softmax.activate(&logits);
/// // The 'probabilities' now represent the probability distribution of classes.
/// ```
///
/// Note: The derivative of Softmax is not straightforward as it depends on all
/// elements of the output vector. It's typically used in conjunction with a loss
/// function, like cross-entropy, in multi-class classification problems.
pub struct SoftmaxActivationFunction;

impl ActivationFunction<&Vec<f64>, Vec<f64>> for SoftmaxActivationFunction {
    /// Computes the Softmax of a given input vector.
    ///
    /// # Arguments
    ///
    /// * `input` - The input vector for which to compute the Softmax.
    ///
    /// # Returns
    ///
    /// A vector representing the probability distribution resulting from the Softmax function.
    fn activate(&self, input: &Vec<f64>) -> Vec<f64> {
        let max = input.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f64> = input.iter().map(|&x| (x - max).exp()).collect();
        let sum_exps: f64 = exps.iter().sum();
        exps.into_iter().map(|exp| exp / sum_exps).collect()
    }

    /// Softmax function does not have a straightforward derivative like other functions,
    /// as it depends on all the elements of the output vector. This method is a placeholder.
    fn derivate(&self, _: &Vec<f64>) -> Vec<f64> {
        // Placeholder for the derivative. In practice, the derivative is used in the
        // context of a loss function (like cross-entropy) in multi-class classification problems.
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_activate_basic() {
        let softmax = SoftmaxActivationFunction;
        let input = vec![1.0, 2.0, 3.0];
        let output = softmax.activate(&input);

        let expected_sum: f64 = output.iter().sum();
        assert!((expected_sum - 1.0).abs() < 1e-5);

        for &val in &output {
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn softmax_activate_extreme() {
        let softmax = SoftmaxActivationFunction;
        let input = vec![1000.0, 1000.0];
        let output = softmax.activate(&input);

        // In extreme cases, values should be approximately equal (due to exp scaling)
        assert!((output[0] - 0.5).abs() < 1e-5);
        assert!((output[1] - 0.5).abs() < 1e-5);
    }
}
