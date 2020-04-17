//
// Created by jk on 2020/3/18.
//

#ifndef SRC_ALGORITHM_H
#define SRC_ALGORITHM_H


#include "Data.h"
#include "logistic.h"
#include "poisson.h"
#include "coxph.h"
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
    double coef0_init;
    double coef0;
    double loss;
    Eigen::VectorXi A_out;
    int l;
    int model_fit_max;
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

    void update_coef0_init(double coef0){
        this->coef0_init = coef0;
    };

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
    PdasLm(Data &data, unsigned int max_iter = 20) : Algorithm(data, 1, max_iter) {}

//    void update_coef0_init(double coef0){
//        this->coef0 = coef0;
//    }

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
            train_x.row(i) = data.x.row(this->train_mask(i));
            train_y(i) = data.y(this->train_mask(i));
            train_weight(i) = data.weight(this->train_mask(i));
        }

        Eigen::VectorXd beta_est;

        int T0 = this->sparsity_level;

        double max_T=0.0;
        vector<int>E(p);
        for(int k=0;k<=p-1;k++) {
        E[k]=k;
        }
        vector<int>I(p-T0);
        vector<int>A(T0);
        vector<int>J(p-T0);
        vector<int>B(T0);
        Eigen::MatrixXd X_A = Eigen::MatrixXd::Zero(train_n, T0);
        Eigen::MatrixXd X_I = Eigen::MatrixXd::Zero(train_n, p-T0);
        Eigen::VectorXd beta_A = Eigen::VectorXd::Zero(T0);
        Eigen::VectorXd d_I = Eigen::VectorXd::Zero(p-T0);
        beta_est=this->beta_init;
        Eigen::VectorXd d=(train_x.transpose()*(train_y-train_x*beta_est)) /double(train_n);
        Eigen::VectorXd bd=beta_est+d;
        bd=bd.cwiseAbs();
        for(int k=0;k<=T0-1;k++) {
            max_T=bd.maxCoeff(&A[k]);
            bd(A[k])=0.0;
        }
        sort (A.begin(),A.end());
        set_difference(E.begin(),E.end(), A.begin(),A.end(),I.begin());
        for(this->l=1;this->l<=this->max_iter;l++) {
            for(int mm=0;mm<=T0-1;mm++) {
              X_A.col(mm)=train_x.col(A[mm]);
            }
            for(int mm=0;mm<=p-T0-1;mm++) {
              X_I.col(mm)=train_x.col(I[mm]);
            }
            beta_A=X_A.colPivHouseholderQr().solve(train_y);
            for(int mm=0;mm<=T0-1;mm++) {
              d(A[mm])=0.0;
              beta_est(A[mm])=beta_A(mm);
            }
            d_I=(X_I.transpose()*(train_y-X_A*beta_A))/double(train_n);
            for(int mm=0;mm<=p-T0-1;mm++) {
              beta_est(I[mm])=0.0;
              d(I[mm])=d_I(mm);
            }
            bd=beta_est+d;
            bd=bd.cwiseAbs();
            for(int k=0;k<=T0-1;k++) {
              max_T=bd.maxCoeff(&B[k]);
              bd(B[k])=0;
            }
            sort (B.begin(),B.end());
            set_difference(E.begin(),E.end(), B.begin(),B.end(),J.begin());
            if(A==B) { break;} else {
              A=B;
              I=J;
            }
        }
        this->beta = beta_est;
//        this->loss = (data.y-data.x*beta_est).squaredNorm()/double(data.get_n());
    };
};

class PdasLogistic : public Algorithm {
public:
    PdasLogistic(Data &data, unsigned int max_iter = 20, int model_fit_max = 1e6) : Algorithm(data, 2, max_iter) {
        this->model_fit_max = model_fit_max;
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

        double weight_mean = train_weight.sum() / train_weight.size();

        for (i = 0; i < train_n; i++) {
            train_weight(i) = train_weight(i) / weight_mean;
        }

        Eigen::VectorXd beta_est;

        int T0 = this->sparsity_level;

        vector<int>A(T0);
        vector<int>B(T0);
        Eigen::MatrixXd X_A(train_n, T0+1);
        Eigen::MatrixXd Xsquare(train_n, p);
        X_A.col(0) = Eigen::VectorXd::Ones(train_n);
        Eigen::VectorXd beta_A(T0+1);
        Eigen::VectorXd bd(p);
        Eigen::VectorXd zero = Eigen::VectorXd::Zero(T0+1);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(train_n);
        Eigen::VectorXd coef(train_n);
        for(i=0;i<=train_n-1;i++) {
            coef(i) = this->coef0_init;
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
                X_A.col(mm +1) = train_x.col(A[mm]);
            }
            beta_A = logit_fit(X_A.rightCols(T0), train_y, train_n, T0, train_weight);  //update beta_A
            beta_est = Eigen::VectorXd::Zero(p);
            for(int mm=0;mm<T0;mm++) {
                beta_est(A[mm]) = beta_A(mm + 1);
            }
            this->coef0 = beta_A(0);
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
            A_out(i) = A[i];
        }
        this->beta = beta_est;
//        this->loss = -2*(data.weight.array()*(data.y.array()*pr.array().log())+(one-data.y).array()*(one-pr).array().log()).sum();
    };
};

class PdasPoisson : public Algorithm {
public:
    PdasPoisson(Data &data, unsigned int max_iter = 20, int model_fit_max = 1e6) : Algorithm(data, 1, max_iter) {
        this->model_fit_max = model_fit_max;
    };

    void fit() {
//        std::cout<<"Poisson fit"<<endl;
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

//        double weight_sum = train_weight.sum();
//
//        for (i = 0; i < train_n; i++) {
//            train_weight(i) = train_weight(i) / weight_sum;
//        }

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

        Eigen::VectorXd coef(train_n);
        for(i=0;i<=train_n-1;i++) {
            coef(i) = this->coef0_init;
        }
//        std::cout<<"Poisson fit 2"<<endl;

        Eigen::VectorXd xbeta_exp = train_x*this->beta_init+coef;
        for(i=0;i<=train_n-1;i++) {
            if(xbeta_exp(i)>30.0) xbeta_exp(i) = 30.0;
            if(xbeta_exp(i)<-30.0) xbeta_exp(i) = -30.0;
        }
        xbeta_exp = xbeta_exp.array().exp();

        Eigen::VectorXd res = train_y-xbeta_exp;
        Eigen::VectorXd g(p);
        for(i=0;i<p;i++){
            g(i) = -res.dot(train_x.col(i));
        }

//        std::cout<<"Poisson fit 3"<<endl;

        Xsquare = train_x.array().square();
//        for(int i=0;i<train_n<i++)
//        {
//            for(int j=0;j<p;j++)
//            {
//                std::cout<<"Poisson fit 3"<<endl;
//            }
//        }

        Eigen::VectorXd h(p);
        for(i=0;i<p;i++){
            h(i) = xbeta_exp.dot(Xsquare.col(i));
        }
//        std::cout<<"Poisson fit 4"<<endl;
        bd = h.cwiseProduct((this->beta_init - g.cwiseQuotient(h)).cwiseAbs2());
//        std::cout<<"Poisson fit 5"<<endl;
        for(int k=0;k<=T0-1;k++) {
            bd.maxCoeff(&A[k]);
            bd(A[k]) = 0.0;
        }
        sort(A.begin(),A.end());
//        std::cout<<"Poisson fit 6"<<endl;


        for(this->l=1;this->l<=this->max_iter;this->l++) {
            for(int mm=0;mm<T0;mm++) {
                X_A.col(mm) = train_x.col(A[mm]);
            }
//            std::cout<<"Poisson fit 2"<<endl;
            beta_A = poisson_fit(X_A, train_y, train_n, T0, train_weight);  //update beta_A
            beta_est = Eigen::VectorXd::Zero(p);
            for(int mm=0;mm<T0;mm++) {
                beta_est(A[mm]) = beta_A(mm);
            }
            this->coef0 = beta_A(T0);
//            std::cout<<"Poisson fit 2.5"<<endl;

            xbeta_exp = X_A*beta_A;
            for(i=0;i<=train_n-1;i++) {
                if(xbeta_exp(i)>30.0) xbeta_exp(i) = 30.0;
                if(xbeta_exp(i)<-30.0) xbeta_exp(i) = -30.0;
            }
            xbeta_exp = xbeta_exp.array().exp();
            res = train_y-xbeta_exp;
//            std::cout<<"Poisson fit 3"<<endl;
            for(i=0;i<p;i++){
                g(i) = -res.dot(train_x.col(i));
            }
            Xsquare = train_x.array().square();
            for(i=0;i<p;i++){
                h(i) = xbeta_exp.dot(Xsquare.col(i));
            }
            bd = h.cwiseProduct((beta_est - g.cwiseQuotient(h)).cwiseAbs2());
//            std::cout<<"Poisson fit 4"<<endl;

//            xbeta_exp = X_A*beta_A;
//            for(i=0;i<=train_n-1;i++) {
//                if(xbeta_exp(i)>25.0) xbeta_exp(i) = 25.0;
//                if(xbeta_exp(i)<-25.0) xbeta_exp(i) = -25.0;
//            }
//            xbeta_exp = xbeta_exp.array().exp();
//            pr = xbeta_exp.array()/(xbeta_exp+one).array();
//            l1 = -train_x.adjoint()*((train_y-pr).cwiseProduct(train_weight));
//            l2 = (Xsquare.adjoint())*((pr.cwiseProduct(one-pr)).cwiseProduct(train_weight));
//            d = -l1.cwiseQuotient(l2);
//            bd = (beta_est+d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
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
//        std::cout<<"Poisson fit end"<<endl;
        this->beta = beta_est;
//        this->loss = -2*(data.weight.array()*(data.y.array()*pr.array().log())+(one-data.y).array()*(one-pr).array().log()).sum();
    };
};

class PdasCox : public Algorithm {
public:
    PdasCox(Data &data, unsigned int max_iter = 20, int model_fit_max = 1e6) : Algorithm(data, 4, max_iter) {
        this->model_fit_max = model_fit_max;
    };

    void fit() {
//        cout<<"cox_fit"<<endl;
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

        double weight_mean = train_weight.sum() / train_weight.size();

        for (i = 0; i < train_n; i++) {
            train_weight(i) = train_weight(i) / weight_mean;
        }

        Eigen::VectorXd beta_est = Eigen::VectorXd::Zero(p);;

        int T0 = this->sparsity_level;

        double max_T=0.0;
        Eigen::VectorXi A = Eigen::VectorXi::Zero(T0);
        Eigen::VectorXi I = Eigen::VectorXi::Zero(p - T0);

        Eigen::MatrixXi A_list(T0, max_iter +2);
        A_list.col(0) = Eigen::VectorXi::Zero(T0);

//        A = getcox_A(train_x,this->beta_init,T0,Eigen::VectorXi::Zero(p),train_y,train_weight);
        getcox_A(train_x,this->beta_init,T0,Eigen::VectorXi::Zero(p),train_y,train_weight, A, I);
//        cout<<1<<endl;
//        for(int mm=0;mm<T0;mm++) {
//            cout<<A[mm]<<" ";
//        }
//        cout<<endl;
        A_list.col(1) = A;

        Eigen::MatrixXd X_A(train_n, T0);
        Eigen::VectorXd beta_A(T0);
        for(this->l=1;this->l<=this->max_iter;this->l++) {
            for(int mm=0;mm<T0;mm++) {
                X_A.col(mm) = train_x.col(A[mm]);
            }
            beta_A = cox_fit(X_A, train_y, train_n, T0, train_weight);  //update beta_A
            for(int mm=0;mm<p - T0;mm++) {
                beta_est(I[mm]) = 0;
            }
            for(int mm=0;mm<T0;mm++) {
                beta_est(A[mm]) = beta_A(mm);
//                cout<<beta_A(mm)<<" ";
            }
//            cout<<endl;
//            this->coef0 = beta_A(0);

//            A = getcox_A(train_x,beta_est,T0,A_list.col(this->l),train_y,train_weight);
            getcox_A(train_x,beta_est,T0,A_list.col(this->l),train_y,train_weight, A, I);
            A_list.col(this->l+1) = A;
            if(A==A_list.col(this->l)|| A==A_list.col(this->l - 1)) break;
        }
        for(i=0;i<T0;i++){
            this->A_out(i) = A[i];
        }
        this->beta = beta_est;
//        this->loss = -2*(data.weight.array()*(data.y.array()*pr.array().log())+(one-data.y).array()*(one-pr).array().log()).sum();
    };
};


class GroupPdasLm : public Algorithm {
public:
    GroupPdasLm(Data &data, unsigned int max_iter = 100) : Algorithm(data, 1, max_iter) {}

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
    GroupPdasLogistic(Data &data, unsigned int max_iter = 100, int model_fit_max = 20) : Algorithm(data, 2, max_iter) {
        this->model_fit_max = model_fit_max;
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
            beta_A = logit_fit(X_A, train_y, train_n, T0, train_weight);  //update beta_A
            beta_est = Eigen::VectorXd::Zero(p);
            for(int mm=0;mm<T0;mm++) {
                beta_est(A[mm]) = beta_A(mm + 1);
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

class GroupPdasPoisson : public Algorithm {
public:
    GroupPdasPoisson(Data &data, unsigned int max_iter = 20, int model_fit_max = 20) : Algorithm(data, 1, max_iter) {
        this->model_fit_max = model_fit_max;
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

//        double weight_sum = train_weight.sum();
//
//        for (i = 0; i < train_n; i++) {
//            train_weight(i) = train_weight(i) / weight_sum;
//        }

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

        Eigen::VectorXd coef(train_n);
        for(i=0;i<=train_n-1;i++) {
            coef(i) = this->coef0;
        }

        Eigen::VectorXd xbeta_exp = train_x*this->beta_init+coef;
        for(i=0;i<=train_n-1;i++) {
            if(xbeta_exp(i)>30.0) xbeta_exp(i) = 30.0;
            if(xbeta_exp(i)<-30.0) xbeta_exp(i) = -30.0;
        }
        xbeta_exp = xbeta_exp.array().exp();

        Eigen::VectorXd res = train_y-xbeta_exp;
        Eigen::VectorXd g(p);
        for(i=0;i<p;i++){
            g(i) = -res.dot(train_x.col(i));
        }

        Xsquare = train_x.array().square();
        Eigen::VectorXd h(p);
        for(i=0;i<p;i++){
            h(i) = xbeta_exp.dot(Xsquare.col(i));
        }

        Eigen::VectorXd b = this->beta_init - g.cwiseQuotient(h);

        bd = h.cwiseProduct(b.cwiseAbs2());

        for(int k=0;k<=T0-1;k++) {
            bd.maxCoeff(&A[k]);
            bd(A[k]) = 0.0;
        }
        sort(A.begin(),A.end());
        for(this->l=1;this->l<=this->max_iter;this->l++) {
            for(int mm=0;mm<T0;mm++) {
                X_A.col(mm) = train_x.col(A[mm]);
            }
            beta_A = poisson_fit(X_A, train_y, train_n, this->sparsity_level, train_weight);  //update beta_A
            beta_est = Eigen::VectorXd::Zero(p);
            for(int mm=0;mm<T0;mm++) {
                beta_est(A[mm]) = beta_A(mm);
            }
            this->coef0 = beta_A(T0);

            xbeta_exp = X_A*beta_A;
            for(i=0;i<=train_n-1;i++) {
                if(xbeta_exp(i)>30.0) xbeta_exp(i) = 30.0;
                if(xbeta_exp(i)<-30.0) xbeta_exp(i) = -30.0;
            }
            xbeta_exp = xbeta_exp.array().exp();
            res = train_y-xbeta_exp;
            for(i=0;i<p;i++){
                g(i) = -res.dot(train_x.col(i));
            }
            Xsquare = train_x.array().square();
            for(i=0;i<p;i++){
                h(i) = xbeta_exp.dot(Xsquare.col(i));
            }
            bd = h.cwiseProduct((beta_est - g.cwiseQuotient(h)).cwiseAbs2());

//            xbeta_exp = X_A*beta_A;
//            for(i=0;i<=train_n-1;i++) {
//                if(xbeta_exp(i)>25.0) xbeta_exp(i) = 25.0;
//                if(xbeta_exp(i)<-25.0) xbeta_exp(i) = -25.0;
//            }
//            xbeta_exp = xbeta_exp.array().exp();
//            pr = xbeta_exp.array()/(xbeta_exp+one).array();
//            l1 = -train_x.adjoint()*((train_y-pr).cwiseProduct(train_weight));
//            l2 = (Xsquare.adjoint())*((pr.cwiseProduct(one-pr)).cwiseProduct(train_weight));
//            d = -l1.cwiseQuotient(l2);
//            bd = (beta_est+d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
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

class GroupPdasCox : public Algorithm {
public:
    GroupPdasCox(Data &data, unsigned int max_iter = 20, int model_fit_max = 1e6) : Algorithm(data, 4, max_iter) {
        this->model_fit_max = model_fit_max;
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

        double weight_mean = train_weight.sum() / train_weight.size();

        for (i = 0; i < train_n; i++) {
            train_weight(i) = train_weight(i) / weight_mean;
        }

        Eigen::VectorXd beta_est;

        int T0 = this->sparsity_level;

        double max_T=0.0;
        Eigen::VectorXi A = Eigen::VectorXi::Zero(T0);
        Eigen::VectorXi I = Eigen::VectorXi::Zero(p - T0);
        Eigen::VectorXi zero = Eigen::VectorXi::Zero(p);

        Eigen::MatrixXi A_list(T0, max_iter +2);
        A_list.col(0) = Eigen::VectorXi::Zero(T0);

        getcox_A(train_x,this->beta_init,T0,zero,train_y,train_weight, A, I);
//        A = getcox_A(train_x,this->beta_init,T0,zero,train_y,train_weight);

        A_list.col(1) = A;

        Eigen::MatrixXd X_A(train_n, T0);
        Eigen::VectorXd beta_A(T0);
        for(this->l=1;this->l<=this->max_iter;this->l++) {
            for(int mm=0;mm<T0;mm++) {
                X_A.col(mm) = train_x.col(A[mm]);
            }
            beta_A = cox_fit(X_A.rightCols(T0), train_y, train_n, T0, train_weight);  //update beta_A
            for(int mm=0;mm<p - T0;mm++) {
                beta_est(I[mm]) = 0;
            }
            for(int mm=0;mm<T0;mm++) {
                beta_est(A[mm]) = beta_A(mm);
            }
//            this->coef0 = beta_A(0);

            getcox_A(train_x,beta_est,T0,A_list.col(this->l),train_y,train_weight, A, I);
//            A = getcox_A(train_x,beta_est,T0,A_list.col(this->l),train_y,train_weight);
            A_list.col(this->l+1) = A;
            if(A==A_list.col(this->l)|| A==A_list.col(this->l - 1)) break;
        }
        for(i=0;i<T0;i++){
            this->A_out(i) = A[i];
        }
        this->beta = beta_est;
//        this->loss = -2*(data.weight.array()*(data.y.array()*pr.array().log())+(one-data.y).array()*(one-pr).array().log()).sum();
    };
};


#endif //SRC_ALGORITHM_H
