#include <RcppArmadillo.h>
using namespace Rcpp;

// Declare the exported function

// [[Rcpp::export]]
List IRLS_pois(const arma::colvec& y,const arma::mat& X,int max_iter = 10000,double tol = 0.0001){
  
  int n = X.n_rows; // Number of observations
  int p = X.n_cols; // Number of covariates
  
  // Initial values
  arma::vec beta = arma::zeros(p);
  arma::vec beta_old = beta;
  
  // Loop
  for(int iter=0; iter < max_iter; ++iter){
    arma::vec eta = X * beta;
    arma::vec mu = arma::exp(eta);
    
    arma::vec w = mu;
    arma::vec z = eta + (y-mu)/mu;
    
    //WLS
    arma::mat W = arma::diagmat(w);
    arma::mat X_t_W = X.t() * W;
    arma::vec R = X_t_W * z;
    arma::mat L = X_t_W * X;
    beta = arma::solve(L,R);
    
    //convergence
    if(arma::norm(beta-beta_old,2)<tol){
      break;
    }
    beta_old = beta;
  }
  
  return List::create(
    Named("coefficients") = beta
  );
  
}

