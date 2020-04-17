#ifndef logistic_H
#define logistic_H

// [[Rcpp::export]]
Eigen::VectorXd pi(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd coef, int n);

Eigen::VectorXd logit_fit(Eigen::MatrixXd x, Eigen::VectorXd y, int n, int p, Eigen::VectorXd weights);

// [[Rcpp::export]]
double loglik_logit(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd coef, int n, Eigen::VectorXd weights);


#endif
