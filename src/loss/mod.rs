//! This module contains implementations for popular Loss Functions

use std::sync::Arc;
use anyhow::Result;

pub mod mse;
pub mod mae;

/// The `LossFunction` trait defines a common interface for loss functions
/// in machine learning algorithms. It is designed to compute a loss metric
/// based on predictions and actual target values.
///
/// This trait is generic over the type `T`, allowing it to be used with
/// different numeric types (e.g., `f32`, `f64`). The use of `Arc<[T]>` for
/// both predictions and targets facilitates safe, concurrent access to these
/// arrays, making the trait suitable for use in multi-threaded contexts.
///
/// The method `compute` returns a `Result<T, anyhow::Error>`, providing a 
/// convenient way to handle errors that might occur during the computation
/// of the loss. This approach leverages the `anyhow` crate for simplified
/// error handling.
/// 
/// # Type Parameters
///
/// - `T`: The type of the elements in the prediction and target arrays. This type
///        should be a numeric type (like `f32` or `f64`) that supports the operations
///        necessary for computing the loss. It must also implement the `Clone` trait
///        to enable efficient sharing of data.
/// 
/// # Example
///
/// Implementing the `LossFunction` trait for Mean Squared Error (MSE):
///
/// ```
/// use std::sync::Arc;
/// use anyhow::Result;
/// use qmachina::loss::LossFunction;
///
/// struct MeanSquaredError;
///
/// impl LossFunction<f64> for MeanSquaredError {
///     fn compute(&self, predictions: Arc<[f64]>, targets: Arc<[f64]>) -> Result<f64> {
///         if predictions.len() != targets.len() {
///             return Err(anyhow::anyhow!("Predictions and targets must have the same length"));
///         }
///         let mse = predictions.iter()
///             .zip(targets.iter())
///             .map(|(p, t)| (p - t).powi(2))
///             .sum::<f64>() / predictions.len() as f64;
///         Ok(mse)
///     }
/// }
/// ```
///
/// # Errors
///
/// This trait method may return an `Err` variant, encapsulated in `anyhow::Error`,
/// to indicate various failure conditions, such as mismatched lengths of prediction
/// and target arrays.
///
/// # Panics
///
/// Implementors should ensure that the method does not panic under normal operation.
/// However, certain conditions, like out-of-memory errors, may still lead to panics.
pub trait LossFunction<T> {
    /// Computes the loss value based on the provided predictions and target values.
    ///
    /// # Parameters
    ///
    /// * `predictions` - An `Arc<[T]>` containing predicted values from the model.
    /// * `targets` - An `Arc<[T]>` containing the actual target values to compare against.
    ///
    /// # Returns
    ///
    /// A `Result<T, anyhow::Error>`, where the `Ok` variant contains the computed loss
    /// value and the `Err` variant encapsulates any errors that occurred during the computation.
    fn compute(&self, predictions: Arc<[T]>, targets: Arc<[T]>) -> Result<T>;
}
