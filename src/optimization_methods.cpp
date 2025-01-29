#include <RcppArmadillo.h>
using namespace Rcpp;

// Declare the exported function

// [[Rcpp::export]]
List gradient_descent_lsq(const arma::colvec& y, const arma::mat& A, arma::colvec x0,
                          double lambda, double gamma, double tol = 0.0001, int max_iter = 10000,
                          bool printing = false) {
  int n = y.n_elem;
  int p = A.n_cols;
  arma::colvec x = x0;
  arma::mat AA = A.t() * A;
  arma::colvec Ay = A.t() * y;
  arma::colvec grad = AA * x0 - Ay + 2 * lambda * x0;
  double prev_loss = 0.5 * arma::as_scalar((y - A * x).t() * (y - A * x)) + lambda * arma::dot(x, x);
  double diff = tol + 1;
  int iter = 0;
  arma::vec diff_rec(max_iter);
  arma::vec loss_rec(max_iter);
  
  while (iter < max_iter && diff > tol) {
    x -= gamma * grad;
    grad = AA * x - Ay + 2 * lambda * x;
    double loss = 0.5 * arma::as_scalar((y - A * x).t() * (y - A * x)) + lambda * arma::dot(x, x);
    diff = std::abs((prev_loss - loss) / prev_loss);
    diff_rec(iter) = diff;
    loss_rec(iter) = loss;
    prev_loss = loss;
    ++iter;
  }
  
  if (printing) Rcout << "Converged after " << iter << " steps\n";
  return List::create(Named("x") = x, Named("diff") = diff_rec, Named("loss") = loss_rec);
}


// [[Rcpp::export]]
List stochastic_gradient_descent_lsq(const arma::colvec& y, const arma::mat& A, arma::colvec x0,
                                     double lambda, int batch, double initial_step_size = 1, 
                                     double tol = 1E-6, int max_iter = 10000, bool printing = false) {
  int n = y.n_elem;
  int p = A.n_cols;
  arma::colvec x = x0;
  arma::colvec grad;
  arma::vec diff_rec(max_iter);
  arma::vec loss_rec(max_iter);
  double prev_loss = 0.5 * arma::as_scalar((y - A * x).t() * (y - A * x)) + lambda * arma::dot(x, x);
  double diff = tol + 1;
  int iter = 0;
  
  while (iter < max_iter && diff > tol) {
    arma::uvec indices = arma::randperm(n, batch);
    arma::mat Asub = A.rows(indices);
    arma::colvec ysub = y.rows(indices);
    arma::mat AA = Asub.t() * Asub;
    arma::colvec Ay = Asub.t() * ysub;
    
    grad = (AA * x - Ay) / batch + 2 * lambda * x / n;
    x -= (initial_step_size / (iter + 1)) * grad;
    
    double loss = 0.5 * arma::as_scalar((y - A * x).t() * (y - A * x)) + lambda * arma::dot(x, x);
    diff = std::abs((prev_loss - loss) / prev_loss);
    diff_rec(iter) = diff;
    loss_rec(iter) = loss;
    prev_loss = loss;
    ++iter;
  }
  
  if (printing) Rcout << "Converged after " << iter << " steps\n";
  return List::create(Named("x") = x, Named("diff") = diff_rec, Named("loss") = loss_rec);
}
