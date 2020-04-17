#ifndef coxph_H
#define coxph_H

// [[Rcpp::export]]
double loglik_cox(Eigen::MatrixXd X, Eigen::VectorXd status, Eigen::VectorXd beta, Eigen::VectorXd weights);

// [[Rcpp::export]]
Eigen::VectorXd cox_fit(Eigen::MatrixXd X, Eigen::VectorXd status, int n, int p, Eigen::VectorXd weights);

// [[Rcpp::export]]
void getcox_A(Eigen::MatrixXd X, Eigen::VectorXd beta, int T0, Eigen::VectorXi B, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXi &A_out, Eigen::VectorXi &I_out);

#endif
