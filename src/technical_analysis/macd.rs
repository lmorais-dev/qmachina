//! A module for Moving Average Convergence Divergence (MACD) analysis.
//!
//! The MACD is a trend-following momentum indicator that shows the relationship
//! between two moving averages of a security’s price. It is calculated by subtracting
//! the long-term Exponential Moving Average (EMA) from the short-term EMA.
use std::sync::Arc;
use anyhow::anyhow;

use crate::technical_analysis::{Indicator, PeriodIndicator};
use super::ema::ExponentialMovingAverage;

/// Represents the Moving Average Convergence Divergence (MACD) indicator.
///
/// The MACD indicator is used in technical analysis to identify moving averages
/// that indicate a new trend, whether it's bullish or bearish.
///
/// # Fields
///
/// - `slow_ema`: The long-term EMA.
/// - `fast_ema`: The short-term EMA.
///
/// # Examples
///
/// ```
/// use qmachina::technical_analysis::macd::MACD;
///
/// let macd = MACD::new(26, 12);
/// ```
///
/// ```
/// use std::sync::Arc;
/// use qmachina::technical_analysis::{Indicator, macd::MACD};
///
/// let data = Arc::new([
///     10.0, 10.5, 11.0, 10.8, 11.5,
///     10.0, 10.5, 11.0, 10.8, 11.5,
///     10.0, 10.5, 11.0, 10.8, 11.5,
///     10.0, 10.5, 11.0, 10.8, 11.5,
///     10.0, 10.5, 11.0, 10.8, 11.5,
///     10.0, 10.5, 11.0, 10.8, 11.5,
///     10.0, 10.5, 11.0, 10.8, 11.5
/// ]);
///
/// let macd = MACD::new(26, 12);
///
/// let result = macd.compute(data);
/// assert!(result.is_ok());
/// ```
pub struct MACD {
    slow_ema: ExponentialMovingAverage,
    fast_ema: ExponentialMovingAverage
}

impl MACD {
    /// Constructs a new `MACD` instance.
    ///
    /// # Arguments
    ///
    /// * `slow_ema_period` - The period for the slow EMA.
    /// * `fast_ema_period` - The period for the fast EMA.
    pub fn new(slow_ema_period: usize, fast_ema_period: usize) -> Self {
        let slow_ema_period = if slow_ema_period == 0 { 1 } else { slow_ema_period };
        let fast_ema_period = if fast_ema_period == 0 { 1 } else { fast_ema_period };

        Self {
            slow_ema: ExponentialMovingAverage::new(slow_ema_period),
            fast_ema: ExponentialMovingAverage::new(fast_ema_period)
        }
    }
}

impl Indicator<f64, f64> for MACD {
    /// Computes the MACD value from a given dataset.
    ///
    /// The MACD is calculated by subtracting the slow EMA from the fast EMA.
    ///
    /// # Arguments
    ///
    /// * `data` - A shared array slice of price data.
    ///
    /// # Returns
    ///
    /// Returns a `Result` which is either the MACD value (`f64`) or an error (`anyhow::Error`).
    ///
    /// # Errors
    ///
    /// This function will return an error if the fast EMA period is not less than the slow EMA period,
    /// or if the length of the data is less than the period of the slow EMA.
    fn compute(&self, data: Arc<[f64]>) -> anyhow::Result<f64> {
        if self.fast_ema.period().ge(&self.slow_ema.period()) {
            return Err(anyhow!("The fast EMA must be less than the slow EMA."));
        }

        if data.len().lt(&self.slow_ema.period()) {
            return Err(anyhow!("Slow EMA period is larger than the Data length."));
        }

        let fast_ema_value = self.fast_ema.compute(data.clone())?;
        let slow_ema_value = self.slow_ema.compute(data.clone())?;

        Ok(fast_ema_value - slow_ema_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn initialization_with_valid_periods() {
        let macd = MACD::new(26, 12);
        assert_eq!(macd.slow_ema.period(), 26);
        assert_eq!(macd.fast_ema.period(), 12);
    }

    #[test]
    #[should_panic(expected = "The fast EMA must be less than the slow EMA.")]
    fn fast_ema_greater_than_slow_panics() {
        let macd = MACD::new(12, 26);
        let data = Arc::new([10.0, 10.5, 11.0, 10.8, 11.5]);
        let _ = macd.compute(data).unwrap();
    }

    #[test]
    fn computes_successfully_with_sufficient_data() {
        let macd = MACD::new(26, 12);
        let data = Arc::new([10.0, 10.5, 11.0, 10.8, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0]);
        assert!(macd.compute(data).is_ok());
    }

    #[test]
    fn error_on_insufficient_data_length() {
        let macd = MACD::new(26, 12);
        let data = Arc::new([10.0, 10.5, 11.0]);
        assert!(macd.compute(data).is_err());
    }
}
