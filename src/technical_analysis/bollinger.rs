//! This module contains definitions and implementations for Bollinger Bands.
//!
//! Bollinger Bands are a type of price envelope developed by John Bollinger.
//! They consist of a middle band being an N-period simple moving average (SMA),
//! an upper band at K times an N-period standard deviation above the middle band,
//! and a lower band at K times an N-period standard deviation below the middle band.
//!
//! Typically, the middle band is the 20-day SMA and the standard deviation is set to 2.

use std::sync::Arc;
use anyhow::{Result, anyhow};

use crate::technical_analysis::{Indicator, PeriodIndicator};
use super::sma::SimpleMovingAverage;

/// Represents Bollinger Bands indicator.
///
/// Bollinger Bands are calculated using a moving average of the closing prices,
/// and then the standard deviation of the closing prices is calculated and multiplied
/// by a specified number (usually 2). The upper and lower bands are then set this number
/// of standard deviations above and below the moving average.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use qmachina::technical_analysis::bollinger::BollingerBands;
/// use qmachina::technical_analysis::Indicator;
///
/// let bb = BollingerBands::new(5);
/// let data = Arc::new([100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0]); // Sample data
/// let (upper_band, lower_band) = bb.compute(data).unwrap();
/// ```
pub struct BollingerBands {
    period: usize,
    sma: SimpleMovingAverage
}

impl BollingerBands {
    /// Constructs a new `BollingerBands` with the given period.
    ///
    /// # Parameters
    ///
    /// * `period` - The period over which the SMA and standard deviation are calculated.
    pub fn new(period: usize) -> Self {
        let period = if period == 0 { 1 } else { period };
        let sma = SimpleMovingAverage::new(period);

        Self {
            period,
            sma
        }
    }
}

impl Indicator<f64, (f64, f64)> for BollingerBands {
    /// Computes the Bollinger Bands values using an `Arc<[f64]>` as input data.
    ///
    /// # Parameters
    ///
    /// * `data` - An `Arc<[f64]>` containing the closing prices for the calculation.
    ///
    /// # Returns
    ///
    /// Returns `Ok((f64, f64))` containing the upper and lower bands, or an error if the calculation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the length of the data is less than the specified period.
    fn compute(&self, data: Arc<[f64]>) -> Result<(f64, f64)> {
        if data.len() < self.period {
            return Err(anyhow!("Data length is less than the period."));
        }

        let sma_value = self.sma.compute(data.clone())?;

        let variance: f64 = data.iter()
            .take(self.period)
            .map(|&value| {
                let diff = value - sma_value;
                diff * diff
            })
            .sum::<f64>() / self.period as f64;

        let std_dev = variance.sqrt();

        let upper_band = sma_value + 2.0 * std_dev;
        let lower_band = sma_value - 2.0 * std_dev;

        Ok((upper_band, lower_band))
    }
}

impl PeriodIndicator for BollingerBands {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn bollinger_bands_creation_with_valid_period() {
        let bb = BollingerBands::new(20);
        assert_eq!(bb.period(), 20, "Bollinger Bands period should be 20");
    }

    #[test]
    fn compute_sufficient_data_arc() {
        let bb = BollingerBands::new(5);
        let data = Arc::new([100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0]);

        let (upper_band, lower_band) = bb.compute(data).unwrap();

        // Validate the upper and lower band values
        // These values should be calculated based on the expected Bollinger Bands calculation
        assert!(upper_band > lower_band, "Upper band should be greater than lower band");

        let expected_upper_band = 103.63;
        let expected_lower_band = 99.56;

        assert!((upper_band - expected_upper_band).abs() < 0.01, "Upper band should be close to the expected value");
        assert!((lower_band - expected_lower_band).abs() < 0.01, "Lower band should be close to the expected value");
    }

    #[test]
    fn compute_insufficient_data_arc() {
        let bb = BollingerBands::new(20);
        let data = Arc::new([100.0, 101.0, 102.0]);
        let result = bb.compute(data);
        assert!(result.is_err(), "Should return an error due to insufficient data");
    }

    #[test]
    fn compute_with_invalid_data_arc() {
        let bb = BollingerBands::new(5);
        let data = Arc::new([100.0, f64::NAN, 102.0, 103.0, 104.0]);
        let result = bb.compute(data);
        assert!(result.is_err(), "Should return an error due to invalid (NaN) data");
    }
}
