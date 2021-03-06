//
// Created by jtwok on 2020/3/8.
//

#ifndef BESS_BESS_H
#define BESS_BESS_H

#include <iostream>
#include "List.h"
#include "Data.h"
#include "Algorithm.h"
#include "Metric.h"
#include "path.h"

#include <Eigen/Eigen>

List bess(Eigen::MatrixXd x, Eigen::VectorXd y, int data_type, Eigen::VectorXd weight,
          bool is_normal,
          int algorithm_type, int model_type, int max_iter,
          int path_type, bool is_warm_start,
          int ic_type, bool is_cv, int K,
          double coef0,
          Eigen::VectorXd state,
          Eigen::VectorXi sequence,
          int s_min, int s_max, int K_max, double epsilon);

void pywrap_bess(double* x, int x_row, int x_col, double* y, int y_len, int data_type, double* weight, int weight_len,
                 bool is_normal,
                 int algorithm_type, int model_type, int max_iter,
                 int path_type, bool is_warm_start,
                 int ic_type, bool is_cv, int K,
                 double* state, int state_len,
                 int* sequence, int sequence_len,
                 int s_min, int s_max, int K_max, double epsilon,
                 double* beta_out, int beta_out_len, double* coef0_out, int coef0_out_len, double* train_loss_out, int train_loss_out_len, double* loss_out, int loss_out_len, double* nullloss_out, double* aic_out, int aic_out_len, double* bic_out, int bic_out_len, double* gic_out, int gic_out_len, int* A_out, int A_out_len, int* l_out
                 );

#endif //BESS_BESS_H
