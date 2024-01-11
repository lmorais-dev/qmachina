//! This module contains various technical analysis indicators.
use std::sync::Arc;
use anyhow::Result;

pub mod sma;

/// The `Indicator` trait defines a common interface for technical analysis indicators.
/// It is designed to compute an indicator value based on a given set of data.
///
/// This trait is generic over the types `T` and `V`, allowing it to be used with
/// different data types and output types. The `compute` method returns a `Result<V, anyhow::Error>`,
/// providing a convenient way to handle errors that might occur during the computation
/// of the indicator. This approach leverages the `anyhow` crate for simplified error handling.
///
/// # Type Parameters
///
/// - `T`: The type of the input data for the indicator. This type should be capable
///        of representing the data series used for computation and must support the operations
///        required for the indicator's calculation.
/// - `V`: The type of the output value for the indicator. This type should be a numeric
///        type (like `f64`) that represents the result of the indicator's computation.
///
/// # Example
///
/// Implementing the `Indicator` trait for a Simple Moving Average (SMA):
///
/// ```
/// use std::sync::Arc;
/// use anyhow::Result;
/// use qmachina::technical_analysis::{Indicator, MovingAverageIndicator};
///
/// pub struct SimpleMovingAverage {
///     period: usize
/// };
///
/// impl Indicator<f64, f64> for SimpleMovingAverage {
///     fn compute(&self, data: Arc<[f64]>) -> Result<f64> {
///         if data.len() < self.period {
///             return Err(anyhow::anyhow!("Data length is less than the period."));
///         }
///         let sum: f64 = data.iter().take(self.period).sum();
///         Ok(sum / self.period as f64)
///     }
/// }
/// ```
///
/// # Errors
///
/// This trait method may return an `Err` variant, encapsulated in `anyhow::Error`,
/// to indicate various failure conditions, such as insufficient data length for computation.
///
/// # Panics
///
/// Implementors should ensure that the method does not panic under normal operation.
/// However, certain conditions, like passing an incorrect data type, may still lead to panics.
pub trait Indicator<T, V> {
    /// Computes the value of the indicator based on the provided data.
    ///
    /// This method should encapsulate the core logic of the indicator, processing the input
    /// data and producing a meaningful output that represents the indicator's current state
    /// or measurement.
    ///
    /// # Parameters
    /// * `data`: Input data of type `T`, upon which the indicator calculation is based.
    ///
    /// # Returns
    /// A `Result` wrapping the computed value (`V`) of the indicator, or an error if the 
    /// computation cannot be performed.
    fn compute(&self, data: Arc<[T]>) -> Result<V>;
}

/// The `MovingAverageIndicator` trait extends the functionality of indicators that
/// utilize a moving average. It provides methods to get and set the period of the
/// moving average, which is a common parameter in many technical analysis indicators.
///
/// This trait does not have specific type parameters, as it mainly deals with the
/// period configuration, which is universally represented as a `usize`.
///
/// # Example
///
/// Implementing the `MovingAverageIndicator` trait for a Simple Moving Average (SMA):
///
/// ```
/// use qmachina::technical_analysis::{MovingAverageIndicator};
///
/// pub struct SimpleMovingAverage {
///     period: usize
/// };
///
/// impl MovingAverageIndicator for SimpleMovingAverage {
///     fn period(&self) -> usize {
///         self.period
///     }
///
///     fn set_period(&mut self, period: usize) {
///         self.period = period;
///     }
/// }
/// ```
///
/// # Panics
///
/// Implementors should ensure that methods do not panic under normal operation.
/// Care should be taken to handle edge cases, such as attempting to set a period of zero.
pub trait MovingAverageIndicator {
    fn period(&self) -> usize;
    fn set_period(&mut self, period: usize);
}