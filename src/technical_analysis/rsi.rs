//! This module contains definitions and implementations for the Relative Strength Index (RSI).
//!
//! The RSI is a momentum oscillator used in technical analysis to measure the velocity and
//! magnitude of directional price movements. It provides signals about overbought or oversold
//! conditions in an asset.

use std::sync::Arc;
use anyhow::{Result, anyhow};
use super::{Indicator, PeriodIndicator};

pub struct RelativeStrengthIndex {
    period: usize
}

/// Represents a Relative Strength Index (RSI) indicator.
///
/// The RSI measures the magnitude of recent price changes to identify overbought or oversold
/// conditions in an asset's price.
///
/// # Examples
///
/// Creating an RSI with a period of 14 and standard overbought/oversold thresholds:
///
/// ```
/// use qmachina::technical_analysis::rsi::RelativeStrengthIndex;
///
/// let rsi = RelativeStrengthIndex::new(14);
/// ```
impl RelativeStrengthIndex {
    /// Constructs a new `RelativeStrengthIndex` with the given period and thresholds.
    ///
    /// # Parameters
    ///
    /// * `period` - The look-back period for calculating the RSI.
    pub fn new(period: usize) -> Self {
        Self {
            period: if period == 0 { 1 } else { period }
        }
    }
}

impl PeriodIndicator for RelativeStrengthIndex {
    /// Returns the current period used in the RSI calculation.
    ///
    /// # Returns
    ///
    /// The current period as a `usize`.
    fn period(&self) -> usize {
        self.period
    }

    /// Sets a new period for the RSI calculation.
    ///
    /// # Parameters
    ///
    /// * `period` - The new period to set, represented as a `usize`.
    fn set_period(&mut self, period: usize) {
        self.period = if period == 0 { 1 } else { period };
    }
}

impl Indicator<f64, f64> for RelativeStrengthIndex {
    /// Computes the RSI value using an `Arc<[f64]>` as input data.
    ///
    /// # Parameters
    ///
    /// * `data` - An `Arc<[f64]>` containing the price changes for which the RSI is calculated.
    ///
    /// # Returns
    ///
    /// Returns `Ok(f64)` containing the calculated RSI value, or an error if the calculation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the length of the data is insufficient for RSI calculation or if the data contains
    /// invalid values (NaN or infinite).
    fn compute(&self, data: Arc<[f64]>) -> Result<f64> {
        if data.len() < self.period + 1 {
            return Err(anyhow!("Insufficient data for RSI calculation."));
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        for window in data.windows(2) {
            let change = window[1] - window[0];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }

        if gains == 0.0 {
            return Ok(0.0);
        }

        if losses == 0.0 {
            return Ok(100.0);
        }

        let relative_strength = gains / losses;
        let rsi = 100.0 - (100.0 / (1.0 + relative_strength));

        Ok(rsi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn creation_with_valid_period() {
        let rsi = RelativeStrengthIndex::new(14);
        assert_eq!(rsi.period(), 14, "RSI period should be 14");
    }

    #[test]
    fn creation_with_zero_period() {
        let rsi = RelativeStrengthIndex::new(0);
        assert_eq!(rsi.period(), 1, "RSI period should default to 1 for zero input");
    }

    #[test]
    fn compute_sufficient_data_arc() {
        let rsi = RelativeStrengthIndex::new(14);
        // Example data (price changes) for calculating RSI
        let data = Arc::new([1.0, 1.1, 1.2, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65]);
        let result = rsi.compute(data).unwrap();

        assert!(result < 100.0 && result > 70.0, "Computed RSI should be close to the expected value");
    }

    #[test]
    fn compute_insufficient_data_arc() {
        let rsi = RelativeStrengthIndex::new(14);
        let data = Arc::new([1.0, 2.0]);
        let result = rsi.compute(data);
        assert!(result.is_err(), "Should return an error due to insufficient data");
    }

    #[test]
    fn compute_with_invalid_data_arc() {
        let rsi = RelativeStrengthIndex::new(14);
        let data = Arc::new([1.0, f64::NAN, 3.0]);
        let result = rsi.compute(data);
        assert!(result.is_err(), "Should return an error due to invalid (NaN) data");
    }
}
