//
// Created by Mamba on 2020/2/18.
//

#ifndef SRC_ALGORITHM_H
#define SRC_ALGORITHM_H


#include "Data.h"
#include "logistic.h"
#include <iostream>

using namespace std;

class Algorithm {

public:
    Data data;
    Eigen::VectorXd beta_init;
    int sparsity_level;
    Eigen::VectorXi train_mask;
    int max_iter;
//    bool warm_start;
    Eigen::VectorXd beta;
    double coef0;
    double loss;
    Eigen::VectorXi A_out;
    int l;
    int glm_max;
    int model_type;

    Algorithm() = default;

    Algorithm(Data &data, int model_type, int max_iter = 100) {
        this->data = data;
        this->max_iter = max_iter;
//        this->warm_start = warm_start;
        this->A_out = Eigen::VectorXi::Zero(data.get_p());
        this->model_type = model_type;
    };

    void update_beta_init(Eigen::VectorXd beta_init) {
//        std::cout << "update beta init"<<endl;

        this->beta_init = beta_init;
    };

    virtual void update_coef0_init(double coef0){};

    void update_sparsity_level(int sparsity_level) {
//        std::cout << "update sparsity level" << endl;
        this->sparsity_level = sparsity_level;
    }

    void update_train_mask(Eigen::VectorXi train_mask) {
//        std::cout << "update train mask" << endl;
        this->train_mask = train_mask;
    }

    double get_loss() {
        return this->loss;
    }

    int get_sparsity_level() {
        return this->sparsity_level;
    }

    Eigen::VectorXd get_beta() {
        return this->beta;
    }

    double get_coef0() {
        return this->coef0;
    }

    Eigen::VectorXi  get_A_out() {
        return this->A_out;
    };

    int get_l() {
        return this->l;
    }

//    virtual void fit(Eigen::VectorXd beta_init, double coef0_init, int sparsity_level, Eigen::VectorXi train_mask){};

    virtual void fit(){};

//    virtual void normal_return();
};

class PdasLm : public Algorithm {
public:
    PdasLm(Data &data, unsigned int max_iter = 100) : Algorithm(data, 1, max_iter) {}

    void update_coef0_init(double coef0){
//        std::cout << "update coef0 init" << endl;
        this->coef0 = coef0;
    }

    void fit() {
//        std::cout << "run PDAS_LM 0"<<endl;
        int i;
        int train_n = this->train_mask.size();
        int p = data.get_p();
        Eigen::MatrixXd train_x(train_n, p);
        Eigen::VectorXd train_y(train_n);
        Eigen::VectorXd train_weight(train_n);

//        std::cout << "run PDAS_LM 1"<<endl;

        for (i = 0; i < train_n; i++) {
//            std::cout << "run PDAS_LM 1.5 "<<"i = "<<i<<endl;
            train_x.row(i) = data.x.row(this->train_mask(i));
//            std::cout << "run PDAS_LM 1.6 "<<"i = "<<i<<endl;
            train_y(i) = data.y(this->train_mask(i));
            train_weight(i) = data.weight(this->train_mask(i));
//            std::cout << "run PDAS_LM 1.7 "<<"i = "<<i<<endl;
        }

        double weight_sum = train_weight.sum();

//        std::cout << "run PDAS_LM 2";

        for (i = 0; i < train_n; i++) {
            train_weight(i) = train_weight(i) / weight_sum;
        }

        Eigen::VectorXd beta_est;
        // algorithm implementation:
//        std::cout << "run PDAS_LM";

        int T0 = this->sparsity_level;

        vector<int>A(T0);
        vector<int>B(T0);
        Eigen::MatrixXd X_A(train_n, T0);
        Eigen::VectorXd beta_A(T0);
        Eigen::VectorXd res = (train_y-train_x*this->beta_init)/double(train_n);
        Eigen::VectorXd d(p);
        for(i=0;i<p;i++){
            d(i) = res.dot(train_x.col(i));
        }
        Eigen::VectorXd bd = this->beta_init+d;
        bd = bd.cwiseAbs();
        for(int k=0;k<T0;k++) {             //update A
            bd.maxCoeff(&A[k]);
            bd(A[k]) = 0.0;
        }
        sort(A.begin(),A.end());
//        std::cout << "run 3";
        for(this->l=1; this->l<=this->max_iter; this->l++) {
            for(int mm=0;mm<T0;mm++) {
                X_A.col(mm) = train_x.col(A[mm]);
            }
            beta_A = X_A.colPivHouseholderQr().solve(train_y);  //update beta_A
            beta_est = Eigen::VectorXd::Zero(p);
            for(int mm=0;mm<T0;mm++) {
                beta_est(A[mm]) = beta_A(mm);
            }
            res = (train_y-X_A*beta_A)/double(train_n);
            for(int mm=0;mm<p;mm++){     //update d_I
                bd(mm) = res.dot(train_x.col(mm));
            }
            for(int mm=0;mm<T0;mm++) {
                bd(A[mm]) = beta_A(mm);
            }
            bd = bd.cwiseAbs();
            for(int k=0;k<T0;k++) {
                bd.maxCoeff(&B[k]);
                bd(B[k]) = 0.0;
            }
            sort(B.begin(),B.end());
            if(A==B) break;
            else A = B;
        }
        for(i=0;i<T0;i++){
            this->A_out(i) = A[i] + 1;
        }

        this->beta = beta_est;
//        this->loss = (data.y-data.x*beta_est).squaredNorm()/double(data.get_n());
    };
};

class PdasLogistic : public Algorithm {
public:
    PdasLogistic(Data &data, unsigned int max_iter = 100, double coef0 = 0, int glm_max = 20) : Algorithm(data, 2, max_iter) {
        this->coef0 = coef0;
        this->glm_max = glm_max;
    };

    void update_coef0_init(double coef0) {
        this->coef0 = coef0;
    };

    void fit() {
        int i;
        int train_n = this->train_mask.size();
        int p = data.get_p();
        Eigen::MatrixXd train_x(train_n, p);
        Eigen::VectorXd train_y(train_n);
        Eigen::VectorXd train_weight(train_n);

        for (i = 0; i < train_n; i++) {
            train_x.row(i) = data.x.row(train_mask(i));
            train_y(i) = data.y(train_mask(i));
            train_weight(i) = data.weight(train_mask(i));
        }

        double weight_sum = train_weight.sum();

        for (i = 0; i < train_n; i++) {
            train_weight(i) = train_weight(i) / weight_sum;
        }

        Eigen::VectorXd beta_est;
        // algorithm implementation:
//        std::cout << "run PDAS_GLM";

        int T0 = this->sparsity_level;

        vector<int>A(T0);
        vector<int>B(T0);
        Eigen::MatrixXd X_A(train_n, T0+1);
        Eigen::MatrixXd Xsquare(train_n, p);
        X_A.col(T0) = Eigen::VectorXd::Ones(train_n);
        Eigen::VectorXd beta_A(T0+1);
        Eigen::VectorXd bd(p);
        Eigen::VectorXd zero = Eigen::VectorXd::Zero(T0+1);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(train_n);
        Eigen::VectorXd coef(train_n);
        for(i=0;i<=train_n-1;i++) {
            coef(i) = this->coef0;
        }
        Eigen::VectorXd xbeta_exp = train_x*this->beta_init+coef;
        for(i=0;i<=train_n-1;i++) {
            if(xbeta_exp(i)>25.0) xbeta_exp(i) = 25.0;
            if(xbeta_exp(i)<-25.0) xbeta_exp(i) = -25.0;
        }
        xbeta_exp = xbeta_exp.array().exp();
        Eigen::VectorXd pr = xbeta_exp.array()/(xbeta_exp+one).array();
        Eigen::VectorXd l1 = -train_x.adjoint()*((train_y-pr).cwiseProduct(train_weight));
        Xsquare = train_x.array().square();
        Eigen::VectorXd l2 = (Xsquare.adjoint())*((pr.cwiseProduct(one-pr)).cwiseProduct(train_weight));
        Eigen::VectorXd d = -l1.cwiseQuotient(l2);
        bd = (this->beta_init+d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
        for(int k=0;k<=T0-1;k++) {
            bd.maxCoeff(&A[k]);
            bd(A[k]) = 0.0;
        }
        sort(A.begin(),A.end());
        for(this->l=1;this->l<=this->max_iter;this->l++) {
            for(int mm=0;mm<T0;mm++) {
                X_A.col(mm) = train_x.col(A[mm]);
            }
            beta_A = logistic(X_A, train_y, zero, train_weight, glm_max);  //update beta_A
            beta_est = Eigen::VectorXd::Zero(p);
            for(int mm=0;mm<T0;mm++) {
                beta_est(A[mm]) = beta_A(mm);
            }
            this->coef0 = beta_A(T0);
            xbeta_exp = X_A*beta_A;
            for(i=0;i<=train_n-1;i++) {
                if(xbeta_exp(i)>25.0) xbeta_exp(i) = 25.0;
                if(xbeta_exp(i)<-25.0) xbeta_exp(i) = -25.0;
            }
            xbeta_exp = xbeta_exp.array().exp();
            pr = xbeta_exp.array()/(xbeta_exp+one).array();
            l1 = -train_x.adjoint()*((train_y-pr).cwiseProduct(train_weight));
            l2 = (Xsquare.adjoint())*((pr.cwiseProduct(one-pr)).cwiseProduct(train_weight));
            d = -l1.cwiseQuotient(l2);
            bd = (beta_est+d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
            for(int k=0;k<T0;k++) {
                bd.maxCoeff(&B[k]);
                bd(B[k]) = 0.0;
            }
            sort(B.begin(),B.end());
            if(A==B) break;
            else A = B;
        }
        for(i=0;i<T0;i++){
            A_out(i) = A[i] + 1;
        }
        this->beta = beta_est;
//        this->loss = -2*(data.weight.array()*(data.y.array()*pr.array().log())+(one-data.y).array()*(one-pr).array().log()).sum();
    };
};


class GroupPdasLm : public Algorithm {
public:
    GroupPdasLm(Data &data, unsigned int max_iter = 100) : Algorithm(data, 1, max_iter) {}

    void update_coef0_init(double coef0){
        this->coef0 = coef0;
    }

    void fit() {
        int i;
        int train_n = this->train_mask.size();
        int p = data.get_p();
        Eigen::MatrixXd train_x(train_n, p);
        Eigen::VectorXd train_y(train_n);
        Eigen::VectorXd train_weight(train_n);

        for (i = 0; i < train_n; i++) {
            train_x.row(i) = data.x.row(train_mask(i));
            train_y(i) = data.y(train_mask(i));
            train_weight(i) = data.weight(train_mask(i));
        };

        double weight_sum = train_weight.sum();

        for (i = 0; i < train_n; i++) {
            train_weight(i) = train_weight(i) / weight_sum;
        }

        Eigen::VectorXd beta_est;
        // algorithm implementation:
//        std::cout << "run PDAS_LM";

        int T0 = this->sparsity_level;

        vector<int>A(T0);
        vector<int>B(T0);
        Eigen::MatrixXd X_A(train_n, T0);
        Eigen::VectorXd beta_A(T0);
        Eigen::VectorXd res = (train_y-train_x*this->beta_init)/double(train_n);
        Eigen::VectorXd d(p);
        for(i=0;i<p;i++){
            d(i) = res.dot(train_x.col(i));
        }
        Eigen::VectorXd bd = this->beta_init+d;
        bd = bd.cwiseAbs();
        for(int k=0;k<T0;k++) {             //update A
            bd.maxCoeff(&A[k]);
            bd(A[k]) = 0.0;
        }
        sort(A.begin(),A.end());
//        std::cout << "run 3";
        for(this->l=1; this->l<=this->max_iter; this->l++) {
            for(int mm=0;mm<T0;mm++) {
                X_A.col(mm) = train_x.col(A[mm]);
            }
            beta_A = X_A.colPivHouseholderQr().solve(train_y);  //update beta_A
            beta_est = Eigen::VectorXd::Zero(p);
            for(int mm=0;mm<T0;mm++) {
                beta_est(A[mm]) = beta_A(mm);
            }
            res = (train_y-X_A*beta_A)/double(train_n);
            for(int mm=0;mm<p;mm++){     //update d_I
                bd(mm) = res.dot(train_x.col(mm));
            }
            for(int mm=0;mm<T0;mm++) {
                bd(A[mm]) = beta_A(mm);
            }
            bd = bd.cwiseAbs();
            for(int k=0;k<T0;k++) {
                bd.maxCoeff(&B[k]);
                bd(B[k]) = 0.0;
            }
            sort(B.begin(),B.end());
            if(A==B) break;
            else A = B;
        }
        for(int i=0;i<T0;i++){
            this->A_out(i) = A[i] + 1;
        }

        this->beta = beta_est;
//        this->loss = (data.y-data.x*beta_est).squaredNorm()/double(data.get_n());
    };
};

class GroupPdasLogistic : public Algorithm {
public:
    GroupPdasLogistic(Data &data, unsigned int max_iter = 100, double coef0 = 0, int glm_max = 20) : Algorithm(data, 2, max_iter) {
        this->coef0 = coef0;
        this->glm_max = glm_max;
    };

    void update_coef0_init(double coef0) {
        this->coef0 = coef0;
    };

    void fit() {
        int i;
        int train_n = this->train_mask.size();
        int p = data.get_p();
        Eigen::MatrixXd train_x(train_n, p);
        Eigen::VectorXd train_y(train_n);
        Eigen::VectorXd train_weight(train_n);

        for (i = 0; i < train_n; i++) {
            train_x.row(i) = data.x.row(train_mask(i));
            train_y(i) = data.y(train_mask(i));
            train_weight(i) = data.weight(train_mask(i));
        };

        double weight_sum = train_weight.sum();

        for (i = 0; i < train_n; i++) {
            train_weight(i) = train_weight(i) / weight_sum;
        };

        Eigen::VectorXd beta_est;
        // algorithm implementation:
//        std::cout << "run PDAS_GLM";

        int T0 = this->sparsity_level;

        vector<int>A(T0);
        vector<int>B(T0);
        Eigen::MatrixXd X_A(train_n, T0+1);
        Eigen::MatrixXd Xsquare(train_n, p);
        X_A.col(T0) = Eigen::VectorXd::Ones(train_n);
        Eigen::VectorXd beta_A(T0+1);
        Eigen::VectorXd bd(p);
        Eigen::VectorXd zero = Eigen::VectorXd::Zero(T0+1);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(train_n);
        Eigen::VectorXd coef(train_n);
        for(int i=0;i<=train_n-1;i++) {
            coef(i) = this->coef0;
        }
        Eigen::VectorXd xbeta_exp = train_x*this->beta_init+coef;
        for(int i=0;i<=train_n-1;i++) {
            if(xbeta_exp(i)>25.0) xbeta_exp(i) = 25.0;
            if(xbeta_exp(i)<-25.0) xbeta_exp(i) = -25.0;
        }
        xbeta_exp = xbeta_exp.array().exp();
        Eigen::VectorXd pr = xbeta_exp.array()/(xbeta_exp+one).array();
        Eigen::VectorXd l1 = -train_x.adjoint()*((train_y-pr).cwiseProduct(train_weight));
        Xsquare = train_x.array().square();
        Eigen::VectorXd l2 = (Xsquare.adjoint())*((pr.cwiseProduct(one-pr)).cwiseProduct(train_weight));
        Eigen::VectorXd d = -l1.cwiseQuotient(l2);
        bd = (this->beta_init+d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
        for(int k=0;k<=T0-1;k++) {
            bd.maxCoeff(&A[k]);
            bd(A[k]) = 0.0;
        }
        sort(A.begin(),A.end());
        for(this->l=1;this->l<=this->max_iter;this->l++) {
            for(int mm=0;mm<T0;mm++) {
                X_A.col(mm) = train_x.col(A[mm]);
            }
            beta_A = logistic(X_A, train_y, zero, train_weight, glm_max);  //update beta_A
            beta_est = Eigen::VectorXd::Zero(p);
            for(int mm=0;mm<T0;mm++) {
                beta_est(A[mm]) = beta_A(mm);
            }
            this->coef0 = beta_A(T0);
            xbeta_exp = X_A*beta_A;
            for(int i=0;i<=train_n-1;i++) {
                if(xbeta_exp(i)>25.0) xbeta_exp(i) = 25.0;
                if(xbeta_exp(i)<-25.0) xbeta_exp(i) = -25.0;
            }
            xbeta_exp = xbeta_exp.array().exp();
            pr = xbeta_exp.array()/(xbeta_exp+one).array();
            l1 = -train_x.adjoint()*((train_y-pr).cwiseProduct(train_weight));
            l2 = (Xsquare.adjoint())*((pr.cwiseProduct(one-pr)).cwiseProduct(train_weight));
            d = -l1.cwiseQuotient(l2);
            bd = (beta_est+d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
            for(int k=0;k<T0;k++) {
                bd.maxCoeff(&B[k]);
                bd(B[k]) = 0.0;
            }
            sort(B.begin(),B.end());
            if(A==B) break;
            else A = B;
        }
        for(int i=0;i<T0;i++){
            A_out(i) = A[i] + 1;
        }
        this->beta = beta_est;
//        this->loss = -2*(data.weight.array()*(data.y.array()*pr.array().log())+(one-data.y).array()*(one-pr).array().log()).sum();
    };
};


#endif //SRC_ALGORITHM_H
