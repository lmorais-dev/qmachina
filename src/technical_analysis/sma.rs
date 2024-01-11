//! This module contains definitions and implementations for Simple Moving Average (SMA).
//!
//! The SMA is a commonly used indicator in technical analysis that averages a certain number
//! of past data points to smooth out price data.
use std::{sync::Arc, ops::Div};
use anyhow::{Result, anyhow};

use super::{Indicator, PeriodIndicator};

pub struct SimpleMovingAverage {
    period: usize
}

/// Represents a Simple Moving Average (SMA) indicator.
///
/// The SMA is calculated by averaging a specific number of previous data points. This struct
/// stores the `period` for which the average is calculated.
///
/// # Examples
///
/// Creating an SMA with a period of 5:
///
/// ```
/// use qmachina::technical_analysis::sma::SimpleMovingAverage;
///
/// let sma = SimpleMovingAverage::new(5);
/// ```
impl SimpleMovingAverage {
    /// Constructs a new `SimpleMovingAverage` with the given period.
    ///
    /// # Parameters
    ///
    /// * `period` - The number of data points to include in the moving average calculation.
    pub fn new(period: usize) -> Self {
        Self {
            period: if period == 0 { 1 } else { period }
        }
    }
}

impl PeriodIndicator for SimpleMovingAverage {
    /// Returns the current period used in the SMA calculation.
    ///
    /// # Returns
    ///
    /// The current period as a `usize`.
    fn period(&self) -> usize {
        self.period
    }

    /// Sets a new period for the SMA calculation.
    ///
    /// # Parameters
    ///
    /// * `period` - The new period to set, represented as a `usize`.
    ///
    /// # Panics
    ///
    /// Panics if the `period` is set to 0. A period of 0 is not valid for a moving average.
    fn set_period(&mut self, period: usize) {
        self.period = if period == 0 { 1 } else { period };
    }
}

impl Indicator<f64, f64> for SimpleMovingAverage {
    /// Computes the SMA value using an `Arc<[f64]>` as input data.
    ///
    /// # Parameters
    ///
    /// * `data` - An `Arc<[f64]>` containing the data points for which the SMA is calculated.
    ///
    /// # Returns
    ///
    /// Returns `Ok(f64)` containing the calculated SMA value, or an error if the calculation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the length of the data is less than the SMA period or if the data contains
    /// invalid values (NaN or infinite).
    fn compute(&self, data: Arc<[f64]>) -> Result<f64> {
        if data.len() < self.period {
            return Err(anyhow!("Period is larger than the sampled data."));
        }

        let sum: f64 = data.iter().take(self.period).sum();
        if sum.is_nan() || sum.is_infinite() {
            return Err(anyhow!("Invalid data encountered during calculations."));
        }

        Ok(sum.div(self.period as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn sma_creation_with_valid_period() {
        let sma = SimpleMovingAverage::new(5);
        assert_eq!(sma.period(), 5, "SMA period should be 5");
    }

    #[test]
    fn sma_creation_with_zero_period() {
        let sma = SimpleMovingAverage::new(0);
        assert_eq!(sma.period(), 1, "SMA period should default to 1 for zero input");
    }

    #[test]
    fn compute_sufficient_data_arc() {
        let sma = SimpleMovingAverage::new(3);

        let data = Arc::new([1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = sma.compute(data).unwrap();
        assert_eq!(result, 2.0, "SMA of last 3 values (1, 2, 3) should be 2.0");
    }

    #[test]
    fn compute_insufficient_data_arc() {
        let sma = SimpleMovingAverage::new(5);
        let data = Arc::new([1.0, 2.0]);
        let result = sma.compute(data);
        assert!(result.is_err(), "Should return an error due to insufficient data");
    }

    #[test]
    fn compute_with_invalid_data_arc() {
        let sma = SimpleMovingAverage::new(3);
        let data = Arc::new([1.0, f64::NAN, 3.0]);
        let result = sma.compute(data);
        assert!(result.is_err(), "Should return an error due to invalid (NaN) data");
    }

    #[test]
    fn period_get_set() {
        let mut sma = SimpleMovingAverage::new(3);
        assert_eq!(sma.period(), 3, "Initial period should be 3");

        sma.set_period(10);
        assert_eq!(sma.period(), 10, "Period after set_period should be 10");
    }
}
