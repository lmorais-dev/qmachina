use std::ops::Div;
use std::sync::Arc;
use anyhow::{Result, anyhow};

use super::LossFunction;

/// Represents the Huber Loss function for regression models.
///
/// Huber Loss is often used in regression problems. Compared to mean squared error, 
/// Huber Loss is less sensitive to outliers in the data due to its combination of 
/// squared error and absolute error.
///
/// This struct implements the `LossFunction<f64>` trait, enabling its use in machine learning models
/// for regression tasks.
///
/// # Mathematical Background
///
/// The Huber Loss is defined as:
///
/// \[
/// L_\delta(a) = \begin{cases} 
/// \frac{1}{2} a^2 & \text{for } |a| \leq \delta, \\
/// \delta(|a| - \frac{1}{2} \delta) & \text{otherwise.}
/// \end{cases}
/// \]
///
/// where `a` is the error `prediction - target`, and `\delta` is a threshold parameter.
///
/// # Usage
///
/// `HuberLossFunction` is used in regression tasks, especially when the data contains outliers.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use qmachina::loss::LossFunction;
/// use qmachina::loss::huber::HuberLossFunction;
///
/// let delta = 1.0; // Threshold for Huber loss
/// let huber_loss = HuberLossFunction::new(delta);
/// let predictions = Arc::new([2.3, 1.7, 3.4]); // Predicted values
/// let targets = Arc::new([2.0, 2.0, 3.0]);    // Actual targets
/// let loss = huber_loss.compute(predictions, targets).expect("Failed to compute loss");
/// // 'loss' now contains the Huber loss value
/// ```
pub struct HuberLossFunction {
    delta: f64,
}

impl HuberLossFunction {
    /// Creates a new instance of the Huber Loss Function with a specified delta.
    ///
    /// # Parameters
    ///
    /// * `delta` - The threshold parameter for the Huber loss function.
    pub fn new(delta: f64) -> Self {
        Self { delta }
    }
}

impl LossFunction<f64> for HuberLossFunction {
    /// Computes the Huber loss between predictions and targets.
    ///
    /// The Huber loss combines the advantages of mean squared error and mean absolute error. 
    /// It applies a squared error to small errors and a linear error to large errors, which 
    /// makes it robust to outliers.
    ///
    /// # Parameters
    ///
    /// * `predictions` - An `Arc<[f64]>` containing the predicted values from the model.
    /// * `targets` - An `Arc<[f64]>` containing the actual target values.
    ///
    /// # Returns
    ///
    /// A `Result<f64, anyhow::Error>`, where:
    ///   - The `Ok` variant contains the computed Huber loss.
    ///   - The `Err` variant encapsulates errors that occur during computation, such as
    ///     mismatched lengths of the predictions and targets arrays.
    ///
    fn compute(&self, predictions: Arc<[f64]>, targets: Arc<[f64]>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(anyhow!("Predictions and targets must have the same length"));
        }

        let loss = predictions.iter()
            .zip(targets.iter())
            .fold(0.0, |acc, (&p, &t)| {
                let error = p - t;
                if error.abs() <= self.delta {
                    acc + 0.5 * error.powi(2)
                } else {
                    acc + self.delta * (error.abs() - 0.5 * self.delta)
                }
            })
            .div(predictions.len() as f64);

        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Test Huber loss with small errors (within delta).
    #[test]
    fn huber_loss_small_errors() {
        let delta = 1.0;
        let huber_loss = HuberLossFunction::new(delta);
        let predictions = Arc::new([1.2, 0.9, 1.1]);
        let targets = Arc::new([1.0, 1.0, 1.0]);
        let loss = huber_loss.compute(predictions, targets).unwrap();
        let expected_loss = 0.01; // Calculated manually
        assert!((loss - expected_loss).abs() < 1e-5);
    }

    /// Test Huber loss with large errors (exceeding delta).
    #[test]
    fn huber_loss_large_errors() {
        let delta = 1.0;
        let huber_loss = HuberLossFunction::new(delta);
        let predictions = Arc::new([3.0, 0.0, 4.0]);
        let targets = Arc::new([1.0, 1.0, 1.0]);
        let loss = huber_loss.compute(predictions, targets).unwrap();
        let expected_loss = 1.5; // Calculated manually
        assert!((loss - expected_loss).abs() < 1e-5);
    }

    /// Test Huber loss with varying errors.
    #[test]
    fn huber_loss_varying_errors() {
        let delta = 1.0;
        let huber_loss = HuberLossFunction::new(delta);
        let predictions = Arc::new([1.5, 0.5, 2.0]);
        let targets = Arc::new([1.0, 1.0, 1.0]);
        let loss = huber_loss.compute(predictions, targets).unwrap();
        let expected_loss = 0.25; // Calculated manually
        assert!((loss - expected_loss).abs() < 1e-5);
    }

    /// Test Huber loss with invalid inputs (mismatched array lengths).
    #[test]
    fn huber_loss_invalid_input() {
        let delta = 1.0;
        let huber_loss = HuberLossFunction::new(delta);
        let predictions = Arc::new([1.5, 0.5]);
        let targets = Arc::new([1.0, 1.0, 1.0]);
        let result = huber_loss.compute(predictions, targets);
        assert!(result.is_err());
    }
}
