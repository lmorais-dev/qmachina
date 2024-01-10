use std::ops::Div;
use std::sync::Arc;
use anyhow::{Result, anyhow};

use super::LossFunction;

/// Represents the Binary Cross-Entropy (BCE) loss function for binary classification models.
///
/// Binary Cross-Entropy is a common loss function used in binary classification problems.
/// It measures the performance of a classification model whose output is a probability value between 0 and 1.
///
/// This struct implements the `LossFunction<f64>` trait, enabling its use in machine learning models
/// for binary classification tasks.
///
/// # Mathematical Background
///
/// Binary Cross-Entropy loss calculates the loss for each prediction and averages over all predictions. 
/// It's defined as:
///
/// \[
/// BCE = -\frac{1}{n} \sum_{i=1}^{n} [y_i \cdot \log(p_i) + (1 - y_i) \cdot \log(1 - p_i)]
/// \]
///
/// where `n` is the number of samples, `y_i` is the ith actual target value, and `p_i` is the ith predicted probability.
///
/// # Usage
///
/// `BinaryCrossEntropyLossFunction` is used when the outputs of a model are probabilities, 
/// and the task is to distinguish between two classes. It's particularly useful when the classes 
/// are imbalanced.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use qmachina::loss::LossFunction;
/// use qmachina::loss::bce::BinaryCrossEntropyLossFunction;
///
/// let bce_loss = BinaryCrossEntropyLossFunction;
/// let predictions = Arc::new([0.7, 0.3, 0.9]); // Predicted probabilities
/// let targets = Arc::new([1.0, 0.0, 1.0]);     // Actual labels
/// let loss = bce_loss.compute(predictions, targets).expect("Failed to compute loss");
/// // 'loss' now contains the binary cross-entropy value
/// ```
///
/// Note: It's crucial that the predictions are probabilities (i.e., values between 0 and 1).
pub struct BinaryCrossEntropyLossFunction;

impl LossFunction<f64> for BinaryCrossEntropyLossFunction {
    /// Computes the Binary Cross-Entropy (BCE) loss between predictions and targets.
    ///
    /// Binary Cross-Entropy loss is a widely-used loss function for binary classification tasks.
    /// It calculates the loss by comparing the predicted probability of the positive class 
    /// against the actual binary target (0 or 1).
    ///
    /// # Parameters
    ///
    /// * `predictions` - An `Arc<[f64]>` containing the predicted probabilities from the model.
    ///   Each element should be a probability value between 0 and 1, indicating the likelihood
    ///   of the positive class.
    /// * `targets` - An `Arc<[f64]>` containing the actual binary targets (0 or 1).
    ///
    /// # Returns
    ///
    /// A `Result<f64, anyhow::Error>`, where:
    ///   - The `Ok` variant contains the computed BCE loss. The loss is calculated as the
    ///     average of the BCE for each individual prediction-target pair.
    ///   - The `Err` variant encapsulates errors that occur during computation, such as:
    ///     - Mismatched lengths of the predictions and targets arrays, indicating that each
    ///       prediction does not correspond to a target.
    ///     - Predictions not being valid probabilities (values not in the range [0, 1]).
    ///     - Undefined logarithmic calculations when probabilities are exactly 0 or 1.
    ///
    /// # Notes
    ///
    /// The computation carefully handles edge cases for probabilities (0 and 1) to avoid
    /// NaN values from undefined logarithmic operations. It ensures that the loss calculation
    /// is robust and reliable across various inputs.
    fn compute(&self, predictions: Arc<[f64]>, targets: Arc<[f64]>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(anyhow!("Predictions and targets must have the same length"));
        }

        let bce = predictions.iter()
            .zip(targets.iter())
            .try_fold(0.0, |acc, (&p, &t)| {
                if !(0.0..=1.0).contains(&p) {
                    Err(anyhow!("Predictions must be probabilities (between 0 and 1)"))
                } else if p == 0.0 {
                    if t == 0.0 {
                        Ok(acc)  // log(1 - 0) = 0, so this term contributes 0 to the sum
                    } else {
                        Err(anyhow!("Undefined logarithm for p = 0 with target = 1"))
                    }
                } else if p == 1.0 {
                    if t == 1.0 {
                        Ok(acc)  // log(1) = 0, so this term contributes 0 to the sum
                    } else {
                        Err(anyhow!("Undefined logarithm for p = 1 with target = 0"))
                    }
                } else {
                    Ok(acc - (t * p.ln() + (1.0 - t) * (1.0 - p).ln()))
                }
            })?
            .div(predictions.len() as f64);

        Ok(bce)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Test BCE with valid probabilities.
    /// Expected result is a specific positive loss value.
    #[test]
    fn bce_valid_probabilities() {
        let bce_loss = BinaryCrossEntropyLossFunction;
        let predictions = Arc::new([0.7, 0.3, 0.9]);
        let targets = Arc::new([1.0, 0.0, 1.0]);
        let loss = bce_loss.compute(predictions, targets).unwrap();
        assert!(loss > 0.0);
    }

    /// Test BCE with a prediction outside of probability range.
    /// Expected result is an error.
    #[test]
    fn bce_invalid_probability() {
        let bce_loss = BinaryCrossEntropyLossFunction;
        let predictions = Arc::new([1.5, 0.3, 0.9]); // Invalid probability (>1.0)
        let targets = Arc::new([1.0, 0.0, 1.0]);
        let result = bce_loss.compute(predictions, targets);
        assert!(result.is_err());
    }

    /// Test BCE with predictions and targets of different lengths.
    /// Expected result is an error.
    #[test]
    fn bce_mismatched_lengths() {
        let bce_loss = BinaryCrossEntropyLossFunction;
        let predictions = Arc::new([0.7, 0.3]);
        let targets = Arc::new([1.0, 0.0, 1.0]);
        let result = bce_loss.compute(predictions, targets);
        assert!(result.is_err());
    }

    /// Test BCE with perfect prediction.
    /// Expected result is a loss close to zero.
    #[test]
    fn bce_perfect_prediction() {
        let bce_loss = BinaryCrossEntropyLossFunction;
        let predictions = Arc::new([1.0, 0.0, 1.0]);
        let targets = Arc::new([1.0, 0.0, 1.0]);
        let loss = bce_loss.compute(predictions, targets).unwrap();
        assert!(loss.abs() < 1e-10); // loss should be very close to zero
    }

    /// Test BCE with varying probabilities.
    /// Expected result is a specific positive loss value.
    #[test]
    fn bce_varying_probabilities() {
        let bce_loss = BinaryCrossEntropyLossFunction;
        let predictions = Arc::new([0.8, 0.2, 0.6]);
        let targets = Arc::new([1.0, 0.0, 0.0]);
        let loss = bce_loss.compute(predictions, targets).unwrap();
        let expected_loss = (-((1.0 * 0.8_f64.ln()) + (1.0 - 0.0) * (1.0_f64 - 0.2_f64).ln() + (1.0 - 0.0) * (1.0_f64 - 0.6_f64).ln()) / 3.0) as f64;
        assert_eq!(loss, expected_loss);
    }
}
