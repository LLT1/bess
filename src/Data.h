//
// Created by Mamba on 2020/2/18.
//

#ifndef SRC_DATA_H
#define SRC_DATA_H

#include <Eigen/Eigen>
#include <iostream>
#include <vector>
#include "normalize.h"
using namespace std;
using namespace Eigen;


class Data {

public:
    Eigen::MatrixXd x;
    Eigen::VectorXd y;
    Eigen::VectorXd weight;
    Eigen::VectorXd x_mean;
    Eigen::VectorXd x_norm;
    double y_mean;
    int n;
    int p;
    int data_type;
    bool is_normal;

    Data() = default;

    Data(Eigen::MatrixXd x, Eigen::VectorXd y, int data_type, Eigen::VectorXd weight, bool is_normal) {
        this->x = x;
        this->y = y;
        this->data_type = data_type;
        this->n = x.rows();
        this->p = x.cols();

        this->weight = weight;
        this->is_normal = is_normal;
        this->x_mean = Eigen::VectorXd::Zero(this->p);
        this->x_norm = Eigen::VectorXd::Zero(this->p);

        if(is_normal){
            this->normalize();
        }
    };

    void add_weight() {
        for(int i=0;i<this->n;i++){
            this->x.row(i) = this->x.row(i)*sqrt(this->weight(i));
            this->y(i) = this->y(i)*sqrt(this->weight(i));
        }
    };

    void normalize() {
//        std::cout<<"normalize"<<endl;
        if(this->data_type == 1){
            Normalize(this->x, this->y, this->weight, this->x_mean, this->y_mean, this->x_norm);
        }
        else{
//            std::cout<<"normalize"<<endl;
            Normalize3(this->x, this->weight, this->x_mean, this->x_norm);
        }
//        std::cout<<"normalize end"<<endl;
        // reload this method for different data type
    };

    int get_n() {
        return this->n;
    };

    int get_p() {
        return this->p;
    };

    double get_nullloss() {
        if(this->data_type ==1){
            return this->y.squaredNorm()/double(this->n);
        } else{
            return -2*log(0.5)*this->weight.sum();
        }
    };
};

//class GaussData : public Data {
//public:
//    GaussData(Eigen::MatrixXd &x, Eigen::VectorXd &y):Data(x, y) {}
//};
//
//class BinomialData : public Data {
//public:
//    BinomialData(Eigen::MatrixXd &x, Eigen::VectorXd &y):Data(x, y) {}
//};
//
//class CoxData : public Data {
//    Eigen::VectorXd censore_state;
//
//public:
//    CoxData(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &censore_state):Data(x, y) {
//        this->censore_state = censore_state;
//    }
//};

#endif //SRC_DATA_H
