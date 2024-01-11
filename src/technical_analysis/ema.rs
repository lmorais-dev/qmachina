//! This module contains definitions and implementations for Exponential Moving Average (EMA).
//!
//! The EMA is a type of moving average that places a greater weight and significance
//! on the most recent data points. It's used in technical analysis to smooth out price
//! and data series for trend identification.
use std::sync::Arc;
use anyhow::{Result, anyhow};
use super::{Indicator, PeriodIndicator};

pub struct ExponentialMovingAverage {
    period: usize,
    smoothing: f64
}

/// Represents an Exponential Moving Average (EMA) indicator.
///
/// The EMA is calculated by applying a weighting factor to the most recent data points.
/// This struct stores the `period` and `smoothing` factor for the calculation.
///
/// # Examples
///
/// Creating an EMA with a period of 5:
///
/// ```
/// use qmachina::technical_analysis::ema::ExponentialMovingAverage;
///
/// let ema = ExponentialMovingAverage::new(5);
/// ```
impl ExponentialMovingAverage {
    pub fn new(period: usize) -> Self {
        let period = if period == 0 { 1 } else { period };
        let smoothing = 2.0 / (period as f64 + 1.0);

        Self {
            period,
            smoothing
        }
    }
}

impl Indicator<f64, f64> for ExponentialMovingAverage {
    /// Computes the EMA value using an `Arc<[f64]>` as input data.
    ///
    /// # Parameters
    ///
    /// * `data` - An `Arc<[f64]>` containing the data points for which the EMA is calculated.
    ///
    /// # Returns
    ///
    /// Returns `Ok(f64)` containing the calculated EMA value, or an error if the calculation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the length of the data is less than the EMA period or if the data contains
    /// invalid values (NaN or infinite).
    fn compute(&self, data: Arc<[f64]>) -> Result<f64> {
        if data.len() < self.period {
            return Err(anyhow!("Period is larger than the sampled data."));
        }

        let slice = &data[(data.len() - self.period)..];

        let mut ema = slice[0];
        for &value in &slice[1..] {
            if value.is_nan() || value.is_infinite() {
                return Err(anyhow!("Invalid data encountered during calculations."));
            }
            ema = (value - ema) * self.smoothing + ema;
        }

        Ok(ema)
    }
}

impl PeriodIndicator for ExponentialMovingAverage {
    /// Returns the current period used in the EMA calculation.
    ///
    /// # Returns
    ///
    /// The current period as a `usize`.
    fn period(&self) -> usize {
        self.period
    }

    /// Sets a new period for the EMA calculation.
    ///
    /// # Parameters
    ///
    /// * `period` - The new period to set, represented as a `usize`.
    fn set_period(&mut self, period: usize) {
        self.period = if period == 0 { 1 } else { period };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn ema_creation_with_valid_period() {
        let ema = ExponentialMovingAverage::new(5);
        assert_eq!(ema.period(), 5, "EMA period should be 5");
    }

    #[test]
    fn ema_creation_with_zero_period() {
        let ema = ExponentialMovingAverage::new(0);
        assert_eq!(ema.period(), 1, "EMA period should default to 1 for zero input");
    }

    #[test]
    fn compute_sufficient_data_arc() {
        let ema = ExponentialMovingAverage::new(3);
        let data = Arc::new([1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = ema.compute(data.clone()).unwrap();
        let expected_ema = 4.25;

        assert!((result - expected_ema).abs() < f64::EPSILON, "EMA should be close to the expected value");
    }

    #[test]
    fn compute_insufficient_data_arc() {
        let ema = ExponentialMovingAverage::new(10);
        let data = Arc::new([1.0, 2.0]);
        let result = ema.compute(data);
        assert!(result.is_err(), "Should return an error due to insufficient data");
    }

    #[test]
    fn compute_with_invalid_data_arc() {
        let ema = ExponentialMovingAverage::new(3);
        let data = Arc::new([1.0, f64::NAN, 3.0]);
        let result = ema.compute(data);
        assert!(result.is_err(), "Should return an error due to invalid (NaN) data");
    }

    #[test]
    fn period_get_set() {
        let mut ema = ExponentialMovingAverage::new(3);
        assert_eq!(ema.period(), 3, "Initial period should be 3");

        ema.set_period(10);
        assert_eq!(ema.period(), 10, "Period after set_period should be 10");
    }
}
