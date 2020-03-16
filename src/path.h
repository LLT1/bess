//
// Created by jtwok on 2020/3/8.
//

#ifndef SRC_PATH_H
#define SRC_PATH_H

#include <Eigen/Eigen>

#include "List.h"
#include "Data.h"
#include "Algorithm.h"
#include "Metric.h"

List sequential_path(Data &data, Algorithm* algorithm, Metric* metric, Eigen::VectorXi sequence, bool is_warm_start = "TRUE");
List gs_path(Data &data, Algorithm* algorithm, Metric* metric, int s_min, int s_max, int K_max, double epsilon, bool is_warm_start);

#endif //SRC_PATH_H
