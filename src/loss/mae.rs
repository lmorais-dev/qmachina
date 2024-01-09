use std::ops::Div;
use std::sync::Arc;
use anyhow::{Result, anyhow};

use super::LossFunction;

/// Represents the Mean Absolute Error (MAE) loss function for regression models.
///
/// Mean Absolute Error is a common loss function used in regression problems.
/// It measures the average of the absolute errors—that is, the average 
/// absolute difference between the estimated values (predictions) and the actual value (targets).
///
/// This struct implements the `LossFunction<f64>` trait, enabling its use in various
/// machine learning models where floating-point precision is required.
///
/// # Mathematical Background
///
/// Given a set of predictions and actual target values, MAE is calculated as the average 
/// of the absolute differences between each prediction and its corresponding target. Mathematically,
/// for a set of `n` values, MAE is defined as:
///
/// \[
/// MAE = \frac{1}{n} \sum_{i=1}^{n} |prediction_i - target_i|
/// \]
///
/// where `prediction_i` is the ith predicted value and `target_i` is the ith actual target value.
///
/// # Usage
///
/// The `MeanAbsoluteErrorLossFunction` is typically used during the training of regression models.
/// It provides a measure of how well the model is performing, with lower values indicating a better fit
/// between the model's predictions and the actual data.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use qmachina::loss::LossFunction;
/// use qmachina::loss::mae::MeanAbsoluteErrorLossFunction;
///
/// let mae_loss = MeanAbsoluteErrorLossFunction;
/// let predictions = Arc::new([2.5, 0.0, 2.1, 1.8]);
/// let targets = Arc::new([3.0, -0.5, 2.0, 2.0]);
/// let loss = mae_loss.compute(predictions, targets).expect("Failed to compute loss");
/// // 'loss' now contains the mean absolute error value
/// ```
///
/// Note: Unlike MSE, MAE is not sensitive to outliers as it does not square the differences. 
/// It's used in regression tasks where the target variable is continuous.
pub struct MeanAbsoluteErrorLossFunction;

impl LossFunction<f64> for MeanAbsoluteErrorLossFunction {
    fn compute(&self, predictions: Arc<[f64]>, targets: Arc<[f64]>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(anyhow!("Predictions and targets must have the same length"));
        }

        let mae = predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            .div(predictions.len() as f64);

        Ok(mae)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Test MAE with perfect prediction.
    /// Expected result is zero loss.
    #[test]
    fn mae_perfect_prediction() {
        let mae_loss = MeanAbsoluteErrorLossFunction;
        let predictions = Arc::new([1.0, 2.0, 3.0]);
        let targets = Arc::new([1.0, 2.0, 3.0]);
        let loss = mae_loss.compute(predictions, targets).unwrap();
        assert_eq!(loss, 0.0);
    }

    /// Test MAE with predictions and targets of different lengths.
    /// Expected result is an error.
    #[test]
    fn mae_mismatched_lengths() {
        let mae_loss = MeanAbsoluteErrorLossFunction;
        let predictions = Arc::new([1.0, 2.0]);
        let targets = Arc::new([1.0, 2.0, 3.0]);
        let result = mae_loss.compute(predictions, targets);
        assert!(result.is_err());
    }

    /// Test MAE with positive differences.
    /// Expected result is a positive loss.
    #[test]
    fn mae_positive_difference() {
        let mae_loss = MeanAbsoluteErrorLossFunction;
        let predictions = Arc::new([2.0, 3.0, 4.0]);
        let targets = Arc::new([1.0, 2.0, 3.0]);
        let loss = mae_loss.compute(predictions, targets).unwrap();
        assert!(loss > 0.0);
    }

    /// Test MAE with negative differences.
    /// Expected result is a positive loss.
    #[test]
    fn mae_negative_difference() {
        let mae_loss = MeanAbsoluteErrorLossFunction;
        let predictions = Arc::new([0.0, 1.0, 2.0]);
        let targets = Arc::new([1.0, 2.0, 3.0]);
        let loss = mae_loss.compute(predictions, targets).unwrap();
        assert!(loss > 0.0);
    }

    /// Test MAE with varying values.
    /// This test uses a mix of positive and negative differences.
    /// Expected result is a specific positive loss value.
    #[test]
    fn mae_varying_values() {
        let mae_loss = MeanAbsoluteErrorLossFunction;
        let predictions = Arc::new([1.5, 2.5, 3.5]);
        let targets = Arc::new([1.0, 3.0, 2.0]);
        let loss = mae_loss.compute(predictions, targets).unwrap();
        let expected_loss = ((0.5 + 0.5 + 1.5) / 3.0) as f64;
        assert_eq!(loss, expected_loss);
    }
}
