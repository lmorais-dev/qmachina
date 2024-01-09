use std::sync::Arc;
use anyhow::{Result, anyhow};

use super::LossFunction;

/// Represents the Categorical Cross-Entropy loss function for multi-class classification models.
///
/// Categorical Cross-Entropy is a loss function widely used in multi-class classification problems. 
/// It measures the performance of a classification model whose output is a probability distribution 
/// over multiple classes.
///
/// The loss is computed as the negative sum of the products of the actual values and the logarithm 
/// of the predicted probabilities. This struct is designed for scenarios where both predictions 
/// and targets are represented as a probability distribution across classes, typically using one-hot encoding.
///
/// # Mathematical Background
///
/// For a set of predictions and actual target values, where each element represents the probability
/// for a class, the Categorical Cross-Entropy is defined as:
///
/// \[
/// CCE = -\sum_{i=1}^{N} y_i \cdot \log(p_i)
/// \]
///
/// where `N` is the number of classes, `y_i` is the actual probability for class `i`, and `p_i` is the predicted 
/// probability for class `i`. The values of `y_i` and `p_i` are elements of their respective `Arc<[f64]>` arrays.
///
/// # Example Usage
///
/// ```
/// use std::sync::Arc;
/// use qmachina::loss::LossFunction;
/// use qmachina::loss::cce::CategoricalCrossEntropyLossFunction;
///
/// let cce_loss = CategoricalCrossEntropyLossFunction;
/// let predictions = Arc::new([0.1, 0.7, 0.2]); // Predicted probabilities for 3 classes
/// let targets = Arc::new([0.0, 1.0, 0.0]);     // Actual target in one-hot encoded form
/// let loss = cce_loss.compute(predictions, targets).expect("Failed to compute loss");
/// ```
pub struct CategoricalCrossEntropyLossFunction;

impl LossFunction<f64> for CategoricalCrossEntropyLossFunction {
    /// Computes the Categorical Cross-Entropy loss.
    ///
    /// This method calculates the loss by comparing each predicted probability distribution
    /// against the actual target distribution, both represented as `Arc<[f64]>`.
    ///
    /// # Parameters
    ///
    /// * `predictions` - An `Arc<[f64]>` representing the predicted probabilities for each class.
    ///   It's expected that the sum of probabilities in this distribution equals 1.
    /// * `targets` - An `Arc<[f64]>` representing the actual target distribution, typically in a one-hot encoded format.
    ///
    /// # Returns
    ///
    /// A `Result<f64, anyhow::Error>`, where:
    ///   - The `Ok` variant contains the computed Categorical Cross-Entropy loss, averaged over all classes.
    ///   - The `Err` variant encapsulates errors that occur during computation, such as mismatched lengths or invalid probabilities.
    ///
    /// # Errors
    ///
    /// An error is returned if:
    ///   - The lengths of predictions and targets arrays are different.
    ///   - The predictions contain values outside the range [0, 1].
    fn compute(&self, predictions: Arc<[f64]>, targets: Arc<[f64]>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(anyhow!("Predictions and targets arrays must have the same length"));
        }

        // Categorical Cross-Entropy computation
        let loss = predictions.iter()
            .zip(targets.iter())
            .try_fold(0.0, |acc, (&p, &t)| {
                if p < 0.0 || p > 1.0 {
                    Err(anyhow!("Predictions must be probabilities (between 0 and 1)"))
                } else {
                    // Avoiding computation for log(0), which is undefined
                    let log_p = if p == 0.0 { 0.0 } else { p.ln() };
                    Ok(acc - t * log_p)
                }
            })?;

        Ok(loss / predictions.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Test Categorical Cross-Entropy with valid probability distributions.
    /// Expected result is a specific positive loss value.
    #[test]
    fn cce_valid_probabilities() {
        let cce_loss = CategoricalCrossEntropyLossFunction;
        let predictions = Arc::new([0.1, 0.7, 0.2, 0.0, 0.1, 0.6]);
        let targets = Arc::new([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let loss = cce_loss.compute(predictions, targets).unwrap();
        // The loss should be greater than 0 since the predictions are not perfect
        assert!(loss > 0.0);
    }

    /// Test Categorical Cross-Entropy with mismatched lengths between predictions and targets.
    /// Expected result is an error.
    #[test]
    fn cce_mismatched_lengths() {
        let cce_loss = CategoricalCrossEntropyLossFunction;
        let predictions = Arc::new([0.7, 0.3, 0.0]);
        let targets = Arc::new([1.0, 0.0]);
        let result = cce_loss.compute(predictions, targets);
        assert!(result.is_err());
    }

    /// Test Categorical Cross-Entropy with a prediction outside the probability range.
    /// Expected result is an error.
    #[test]
    fn cce_invalid_probabilities() {
        let cce_loss = CategoricalCrossEntropyLossFunction;
        let predictions = Arc::new([1.5, -0.5, 0.6]); // Invalid probabilities
        let targets = Arc::new([1.0, 0.0, 0.0]);
        let result = cce_loss.compute(predictions, targets);
        assert!(result.is_err());
    }

    /// Test Categorical Cross-Entropy with perfect predictions.
    /// Expected result is a loss close to zero.
    #[test]
    fn cce_perfect_prediction() {
        let cce_loss = CategoricalCrossEntropyLossFunction;
        let predictions = Arc::new([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let targets = Arc::new([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let loss = cce_loss.compute(predictions, targets).unwrap();
        // The loss should be very close to 0 for perfect predictions
        assert!(loss.abs() < 1e-6);
    }
}
