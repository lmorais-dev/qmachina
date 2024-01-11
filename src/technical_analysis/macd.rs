//! A module for Moving Average Convergence Divergence (MACD) analysis.
//!
//! The MACD is a trend-following momentum indicator that shows the relationship
//! between two moving averages of a security’s price. It is calculated by subtracting
//! the long-term Exponential Moving Average (EMA) from the short-term EMA.
use anyhow::{Result, anyhow};

use crate::technical_analysis::{Indicator, PeriodIndicator};
use super::ema::ExponentialMovingAverage;

/// Represents the Moving Average Convergence Divergence (MACD) indicator.
///
/// The MACD is a trend-following momentum indicator used in technical analysis
/// to identify moving averages that signal new trends, whether bullish or bearish.
/// It consists of the MACD line (the difference between two exponential moving averages)
/// and the signal line, which is an EMA of the MACD line.
///
/// # Fields
///
/// * `slow_ema`: The long-term (slow) Exponential Moving Average (EMA).
/// * `fast_ema`: The short-term (fast) Exponential Moving Average (EMA).
/// * `signal_ema`: The EMA of the MACD line, known as the signal line.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use qmachina::technical_analysis::Indicator;
/// use qmachina::technical_analysis::macd::MACD;
///
/// // Create a MACD indicator with specific periods
/// let macd = MACD::new(26, 12, 9);
///
/// // Example data (price values)
/// let data = vec![10.0, 10.5, 11.0, 10.8, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0,
///                 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
///                 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0
/// ];
///
/// // Compute the MACD value
/// let macd_value = macd.compute(&data).expect("Failed to compute MACD");
///
/// let data = vec![10.0, 10.5, 11.0, 10.8, 11.5, 12.0, 12.5, 13.0, 13.5];
/// // Generate the signal line value
/// let signal_value = macd.generate_signal(&data).expect("Failed to compute signal line");
/// ```
pub struct MACD {
    slow_ema: ExponentialMovingAverage,
    fast_ema: ExponentialMovingAverage,
    signal_ema: ExponentialMovingAverage
}

impl MACD {
    /// Constructs a new `MACD` instance.
    ///
    /// # Arguments
    ///
    /// * `slow_ema_period` - The period for the slow EMA.
    /// * `fast_ema_period` - The period for the fast EMA.
    pub fn new(slow_ema_period: usize, fast_ema_period: usize, signal_ema_period: usize) -> Self {
        let slow_ema_period = if slow_ema_period == 0 { 1 } else { slow_ema_period };
        let fast_ema_period = if fast_ema_period == 0 { 1 } else { fast_ema_period };
        let signal_ema_period = if signal_ema_period == 0 { 1 } else { signal_ema_period };

        Self {
            slow_ema: ExponentialMovingAverage::new(slow_ema_period),
            fast_ema: ExponentialMovingAverage::new(fast_ema_period),
            signal_ema: ExponentialMovingAverage::new(signal_ema_period)
        }
    }

    /// Generates the signal line value from a series of MACD values.
    ///
    /// This method computes the signal line, which is an EMA of the MACD line.
    ///
    /// # Arguments
    ///
    /// * `macd_values` - A vector of MACD line values.
    ///
    /// # Returns
    ///
    /// Returns a `Result` which is either the signal line value (`f64`) or an error (`anyhow::Error`).
    ///
    /// # Errors
    ///
    /// Returns an error if the length of the `macd_values` is not equal to the specified `period`.
    pub fn generate_signal(&self, macd_values: &Vec<f64>) -> Result<f64> {
        if macd_values.len() != self.signal_ema.period() {
            return Err(anyhow!("Period is larger or smaller than the Data."));
        }

        let signal_value = self.signal_ema.compute(macd_values)?;

        Ok(signal_value)
    }
}

impl Indicator<f64, f64> for MACD {
    fn compute(&self, data: &Vec<f64>) -> Result<f64> {
        if self.fast_ema.period().ge(&self.slow_ema.period()) {
            return Err(anyhow!("The fast EMA must be less than the slow EMA."));
        }

        if data.len().lt(&self.slow_ema.period()) {
            return Err(anyhow!("Slow EMA period is larger than the Data length."));
        }

        let fast_ema_value = self.fast_ema.compute(data)?;
        let slow_ema_value = self.slow_ema.compute(data)?;

        Ok(fast_ema_value - slow_ema_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialization_with_valid_periods() {
        let macd = MACD::new(26, 12, 9);
        assert_eq!(macd.slow_ema.period(), 26);
        assert_eq!(macd.fast_ema.period(), 12);
    }

    #[test]
    #[should_panic(expected = "The fast EMA must be less than the slow EMA.")]
    fn fast_ema_greater_than_slow_panics() {
        let macd = MACD::new(12, 26, 9);
        let data = vec![10.0, 10.5, 11.0, 10.8, 11.5];
        let _ = macd.compute(&data).unwrap();
    }

    #[test]
    fn computes_successfully_with_sufficient_data() {
        let macd = MACD::new(26, 12, 9);
        let data = vec![10.0, 10.5, 11.0, 10.8, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0,
                        14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
                        20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0];

        assert!(macd.compute(&data).is_ok());
    }

    #[test]
    fn error_on_insufficient_data_length() {
        let macd = MACD::new(26, 12, 9);
        let data = vec![10.0, 10.5, 11.0];
        assert!(macd.compute(&data).is_err());
    }

    #[test]
    fn signal_generation_with_correct_data() {
        let macd = MACD::new(26, 12, 9);
        let macd_values = vec![10.0, 10.5, 11.0, 10.8, 11.5, 12.0, 12.5, 13.0, 13.5];
         // Example period
        let result = macd.generate_signal(&macd_values);
        assert!(result.is_ok());
    }

    #[test]
    fn signal_generation_fails_with_incorrect_period() {
        let macd = MACD::new(26, 12, 9);
        let macd_values = vec![10.0, 10.5, 11.0, 10.8, 11.5, 12.0, 12.5, 13.0];
        let result = macd.generate_signal(&macd_values);
        assert!(result.is_err());
    }

    #[test]
    fn signal_generation_with_empty_macd_values() {
        let macd = MACD::new(26, 12, 9);
        let macd_values = vec![]; // Empty vector
        let result = macd.generate_signal(&macd_values);
        assert!(result.is_err());
    }
}
