//! This module contains various activation functions implementations.

pub mod step;
pub mod sigmoid;
pub mod tanh;
pub mod relu;
pub mod leaky_relu;
pub mod param_relu;
pub mod elu;
pub mod swish;
pub mod softmax;

/// `ActivationFunction` trait defines a general interface for activation functions
/// used in neural networks. Activation functions are fundamental to neural networks
/// as they introduce non-linearity, allowing the network to learn complex patterns
/// and perform tasks beyond just linear classification or regression.
///
/// This trait is generic, allowing for flexibility in the kinds of data structures
/// and types that the activation functions can handle. The generic type parameters
/// `X` and `Y` enable this trait to be implemented for various input and output types,
/// supporting a wide range of neural network architectures and applications.
///
/// # Type Parameters
///
/// * `X`: Represents the type of the input to the activation function. This could be
///   a single value (like `f64`), a complex data structure (like `Vec<f64>`), or any
///   other type that represents the input to a neuron or a layer in a neural network.
///
/// * `Y`: Represents the type of the output from the activation function. Similar to `X`,
///   this could range from a single value to more complex data structures, depending on
///   the design and requirements of the neural network.
///
/// # Implementations
///
/// Implementations of this trait could include standard activation functions like
/// Sigmoid, Tanh, ReLU, and their variants, each potentially tailored to handle
/// different kinds of inputs and outputs as required by specific neural network models.
pub trait ActivationFunction<X, Y> {
    /// Computes the activated value for a given input.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value to the activation function of type `X`.
    ///
    /// # Returns
    ///
    /// Returns the activated output of type `Y`.
    fn activate(&self, input: X) -> Y;

    /// Computes the derivative of the activation function for a given input.
    ///
    /// # Arguments
    ///
    /// * `input` - The input value for which the derivative is to be calculated, of type `X`.
    ///
    /// # Returns
    ///
    /// Returns the derivative of the activation function at the given input, of type `Y`.
    fn derivate(&self, input: X) -> Y;
}
