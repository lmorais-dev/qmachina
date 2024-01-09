use std::ops::Div;

use super::LossFunction;

use anyhow::{Result, anyhow};

/// Represents the Mean Squared Error (MSE) loss function for regression models.
///
/// Mean Squared Error is a common loss function used in regression problems.
/// It measures the average of the squares of the errors—that is, the average squared 
/// difference between the estimated values (predictions) and the actual value (targets).
///
/// This struct implements the `LossFunction<f64>` trait, enabling its use in various
/// machine learning models where floating-point precision is required.
///
/// # Mathematical Background
///
/// Given a set of predictions and actual target values, MSE is calculated as the average 
/// of the squared differences between each prediction and its corresponding target. Mathematically,
/// for a set of `n` values, MSE is defined as:
///
/// \[
/// MSE = \frac{1}{n} \sum_{i=1}^{n} (prediction_i - target_i)^2
/// \]
///
/// where `prediction_i` is the ith predicted value and `target_i` is the ith actual target value.
///
/// # Usage
///
/// The `MeanSquaredErrorLossFunction` is typically used during the training of regression models.
/// It provides a measure of how well the model is performing, with lower values indicating a better fit
/// between the model's predictions and the actual data.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use qmachina::loss::LossFunction;
/// use qmachina::loss::mse::MeanSquaredErrorLossFunction;
///
/// let mse_loss = MeanSquaredErrorLossFunction;
/// let predictions = Arc::new([2.5, 0.0, 2.1, 1.8]);
/// let targets = Arc::new([3.0, -0.5, 2.0, 2.0]);
/// let loss = mse_loss.compute(predictions, targets).expect("Failed to compute loss");
/// // 'loss' now contains the mean squared error value
/// ```
///
/// Note: The MSE loss function is sensitive to outliers as it squares the differences. 
/// It's primarily used in regression tasks where the target variable is continuous.
pub struct MeanSquaredErrorLossFunction;

impl LossFunction<f64> for MeanSquaredErrorLossFunction {
    /// Computes the Mean Squared Error (MSE) between predictions and targets.
    ///
    /// This method calculates the MSE, a common loss function in regression,
    /// by averaging the squares of the differences between each predicted and 
    /// actual target value. It's crucial in evaluating the performance of the 
    /// regression model, with lower values indicating a more accurate model.
    ///
    /// # Parameters
    ///
    /// * `predictions` - An `Arc<[f64]>` representing the predicted values from the model.
    /// * `targets` - An `Arc<[f64]>` representing the actual target values.
    ///
    /// # Returns
    ///
    /// A `Result<f64, anyhow::Error>`, where the `Ok` variant contains the computed MSE,
    /// and the `Err` variant encapsulates errors, primarily when the lengths of predictions
    /// and targets arrays do not match.
    ///
    /// # Errors
    ///
    /// This method returns an error if `predictions` and `targets` have different lengths,
    /// as it's not possible to compute MSE for mismatched data sets.
    fn compute(&self, predictions: std::sync::Arc<[f64]>, targets: std::sync::Arc<[f64]>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(anyhow!("Predictions and targets must have the same length"));
        }

        let mse = predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            .div(predictions.len() as f64);

        Ok(mse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Test MSE with perfect prediction.
    /// Expected result is zero loss.
    #[test]
    fn test_mse_perfect_prediction() {
        let mse_loss = MeanSquaredErrorLossFunction;
        let predictions = Arc::new([1.0, 2.0, 3.0]);
        let targets = Arc::new([1.0, 2.0, 3.0]);
        let loss = mse_loss.compute(predictions, targets).unwrap();
        assert_eq!(loss, 0.0);
    }

    /// Test MSE with predictions and targets of different lengths.
    /// Expected result is an error.
    #[test]
    fn test_mse_mismatched_lengths() {
        let mse_loss = MeanSquaredErrorLossFunction;
        let predictions = Arc::new([1.0, 2.0]);
        let targets = Arc::new([1.0, 2.0, 3.0]);
        let result = mse_loss.compute(predictions, targets);
        assert!(result.is_err());
    }

    /// Test MSE with positive differences.
    /// Expected result is a positive loss.
    #[test]
    fn test_mse_positive_difference() {
        let mse_loss = MeanSquaredErrorLossFunction;
        let predictions = Arc::new([2.0, 3.0, 4.0]);
        let targets = Arc::new([1.0, 2.0, 3.0]);
        let loss = mse_loss.compute(predictions, targets).unwrap();
        assert!(loss > 0.0);
    }

    /// Test MSE with negative differences.
    /// Expected result is a positive loss.
    #[test]
    fn test_mse_negative_difference() {
        let mse_loss = MeanSquaredErrorLossFunction;
        let predictions = Arc::new([0.0, 1.0, 2.0]);
        let targets = Arc::new([1.0, 2.0, 3.0]);
        let loss = mse_loss.compute(predictions, targets).unwrap();
        assert!(loss > 0.0);
    }

    /// Test MSE with varying values.
    /// This test uses a mix of positive and negative differences.
    /// Expected result is a specific positive loss value.
    #[test]
    fn test_mse_varying_values() {
        let mse_loss = MeanSquaredErrorLossFunction;
        let predictions = Arc::new([1.5, 2.5, 3.5]);
        let targets = Arc::new([1.0, 3.0, 2.0]);
        let loss = mse_loss.compute(predictions, targets).unwrap();
        let expected_loss = ((0.5f64.powi(2) + 0.5f64.powi(2) + 1.5f64.powi(2)) / 3.0) as f64;
        assert_eq!(loss, expected_loss);
    }
}
