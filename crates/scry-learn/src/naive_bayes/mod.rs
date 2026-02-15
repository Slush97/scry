//! Naive Bayes classifiers: Gaussian, Bernoulli, and Multinomial.

mod gaussian;
mod bernoulli;
mod multinomial;

pub use gaussian::GaussianNb;
pub use bernoulli::BernoulliNB;
pub use multinomial::MultinomialNB;
