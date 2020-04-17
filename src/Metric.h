//
// Created by Mamba on 2020/2/18.
//

#ifndef SRC_METRICS_H
#define SRC_METRICS_H

#include "Data.h"
#include "Algorithm.h"
#include "coxph.h"
#include <vector>

class Metric {
public:
    bool is_cv;
    int K;
    int ic_type;

    Metric() = default;

    Metric(int ic_type, bool is_cv, int K=0) {
        this->is_cv = is_cv;
        this->ic_type = ic_type;
        this->K = K;
    };

    virtual double test_loss(Algorithm* algorithm, Data &data){};

    virtual double train_loss(Algorithm* algorithm, Data &data){};

    virtual double ic(Algorithm* algorithm, Data &data){};

//    virtual double aic(Algorithm* algorithm, Data &data){};
//    virtual double bic(Algorithm* &algorithm, Data &data){};
//    virtual double gic(Algorithm* &algorithm, Data &data){};
//    virtual double cv(Algorithm* &algorithm, Data &data);
};

class LmMetric : public Metric {
public:

    LmMetric(int ic_type, bool is_cv, int K=0):Metric(ic_type, is_cv, K){};

    double train_loss(Algorithm* algorithm, Data &data){
        return (data.y-data.x * algorithm->get_beta()).array().square().sum()/(2*data.get_n());
    };

    double test_loss(Algorithm* algorithm, Data &data){
        if(!this->is_cv){
            return (data.y-data.x * algorithm->get_beta()).array().square().sum()/(2*data.get_n());
        }
        else{
            int k, i, kk, kkk;
            int n = data.get_n();
            int p = data.get_p();
            int temp, change_index;
            Eigen::VectorXi index_list(n);


            Eigen::VectorXd loss_list(this->K);

            for(i=0;i<n;i++)
            {
                index_list(i) = int(i);
            }
            for (i=n-1;i>0;i--)
            {
                change_index = rand() % int(i+1);
                temp = index_list[i];
                index_list[i] = index_list[change_index];
                index_list[change_index] = temp;
            }

            vector<Eigen::VectorXi> group_list(this->K);
            int group_size = int(n / this->K);
            for(k=0;k<(this->K - 1);k++)
            {
                group_list[k] = index_list.segment(int(k * group_size), group_size);
            }




            group_list[this->K - 1] = index_list.segment(int((this->K - 1) * group_size), n - int(int(this->K - 1) * group_size));

//            for(k=0;k<this->K;k++)
//            {
//                for(kk=0;kk<group_list[k].size();kk++)
//                    std::cout<<group_list[k][kk]<<" ";
//                std::cout<<endl;
//            }


            for(k=0;k<this->K;k++)
            {
                int train_x_size = 0;
                for(kk=0;kk<this->K;kk++)
                {
                    if(kk != k) {
                        train_x_size = train_x_size + group_list[kk].size();
                    }
                }

                // get train_mask

                Eigen::VectorXi train_mask(train_x_size);
                i = 0;
                for(kk=0;kk<this->K;kk++)
                {
                    if(kk != k)
                    {
                        for(kkk=0;kkk<group_list[kk].size();kkk++)
                        {
                            train_mask(i) = group_list[kk](kkk);
                            i++;
                        }
                    }
                }
//                std::cout<<"i = "<<i<<endl;

//                for(i=0;i<train_x_size;i++)
//                {
//                    std::cout<<train_mask(i)<<" ";
//                }

                //get test_x, test_y
                Eigen::MatrixXd test_x(group_list[k].size(), p);
                Eigen::VectorXd test_y(group_list[k].size());

//                std::cout<<"train size"<< train_x_size<<endl;
//                std::cout<<"test size"<< group_list[k].size()<<endl;

                for(i=0;i<group_list[k].size();i++){
                    test_x.row(i) = data.x.row(group_list[k](i));
                    test_y(i) = data.y(group_list[k](i));
                };

                algorithm->update_train_mask(train_mask);
                algorithm->fit();

                loss_list(k) =  (test_y-test_x * algorithm->get_beta()).array().square().sum()/double(2 * group_list[k].size());
            }
//            for(i=0;i<loss_list.size();i++)
//                std::cout<<loss_list(i)<<" "<<endl;
            return loss_list.sum() / double(loss_list.size());
//            std::cout<<"cv_end"<<endl;
        }

    };

    double ic(Algorithm* algorithm, Data &data){
        if(this->is_cv){
            return this->test_loss(algorithm, data);
        }
        else
        {
            if(ic_type == 1){
                return double(data.get_n()) * log(this->train_loss(algorithm, data)) + 2.0 * algorithm->get_sparsity_level();
            } else if (ic_type == 2){
                return double(data.get_n()) * log(this->train_loss(algorithm, data)) + log(double(data.get_n())) * algorithm->get_sparsity_level();
            } else if (ic_type == 3){
                return double(data.get_n()) * log(this->train_loss(algorithm, data)) + log(double(data.get_p())) * log(log(double(data.get_n()))) * algorithm->get_sparsity_level();
            } else if (ic_type == 4){
                return double(data.get_n()) * log(this->train_loss(algorithm, data)) + (log(double(data.get_n())) + 2 * log(double(data.get_p()))) * algorithm->get_sparsity_level();
            }

        }
    };

};

class LogisticMetric : public Metric {
public:

    LogisticMetric(int ic_type, bool is_cv, int K=0):Metric(ic_type, is_cv, K){};

    double train_loss(Algorithm* algorithm, Data &data){
        int i;
        int n = data.get_n();
        Eigen::VectorXd coef(n);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);

        for(i=0;i<=n-1;i++) {
            coef(i) = algorithm->get_coef0();
        }
        Eigen::VectorXd xbeta_exp = data.x*algorithm->get_beta()+coef;
        for(int i=0;i<=n-1;i++) {
            if(xbeta_exp(i)>25.0) xbeta_exp(i) = 25.0;
            if(xbeta_exp(i)<-25.0) xbeta_exp(i) = -25.0;
        }
        xbeta_exp = xbeta_exp.array().exp();
        Eigen::VectorXd pr = xbeta_exp.array()/(xbeta_exp+one).array();
        return -2*(data.weight.array()*((data.y.array()*pr.array().log())+(one-data.y).array()*(one-pr).array().log())).sum();
    }

    double test_loss(Algorithm *algorithm, Data &data) {
        if (!is_cv) {
            return this->train_loss(algorithm, data);
        } else {
//            std::cout<<"cv"<<endl;
            int k, i, kk, kkk;
            int n = data.get_n();
            int p = data.get_p();
            int temp, change_index;
            Eigen::VectorXi index_list(n);
            Eigen::VectorXd loss_list(this->K);

            for (i = 0; i < n; i++) {
                index_list(i) = int(i);
            }
            for (i = n - 1; i > 0; i--) {
                change_index = rand() % int(i + 1);
                temp = index_list[i];
                index_list[i] = index_list[change_index];
                index_list[change_index] = temp;
            }

//            for (i = 0; i < n; i++) {
//                std::cout<<index_list(i)<<" ";
//            }
//            std::cout<<endl;


            vector<Eigen::VectorXi> group_list(this->K);
            int group_size = int(n / this->K);
            for (k = 0; k < (this->K - 1); k++) {
                group_list[k] = index_list.segment(int(k * group_size), group_size);
            }



            group_list[this->K - 1] = index_list.segment(int((this->K - 1) * group_size),
                                                         n - (this->K - 1) * group_size);

//            for(k=0;k<(this->K);k++)
//            {
//                for(kk=0;kk<group_list[k].size();kk++)
//                    std::cout<<group_list[k][kk]<<" ";
//                std::cout<<endl;
//            }
//            std::cout<<"cv_1"<<endl;
            for (k = 0; k < this->K; k++) {
                int train_x_size = 0;
                for (kk = 0; kk < this->K; kk++) {
                    if (kk != k) {
                        train_x_size = train_x_size + group_list[kk].size();
                    }
                }

                // get train_mask
                Eigen::VectorXi train_mask(train_x_size);
                i = 0;
                for (kk = 0; kk < this->K; kk++) {
                    if (kk != k) {
                        for (kkk = 0; kkk < group_list[kk].size(); kkk++) {
                            train_mask[i] = group_list[kk](kkk);
                            i++;
                        }
                    }
                }
//                for(i=0;i<train_x_size;i++)
//                {
//                    std::cout<<train_mask(i)<<" ";
//                }
//                std::cout<<endl;
//
//                std::cout<<"cv_2"<<endl;
//
//                std::cout<<"train size"<< train_x_size<<endl;


                //get test_x, test_y
                int test_n = group_list[k].size();

//                std::cout<<"test size"<<test_n<<endl;
                Eigen::MatrixXd test_x(test_n, p);
                Eigen::VectorXd test_y(test_n);
                Eigen::VectorXd test_weight(test_n);

                for (i = 0; i < group_list[k].size(); i++) {
                    test_x.row(i) = data.x.row(group_list[k](i));
                    test_y(i) = data.y(group_list[k](i));
                    test_weight(i) = data.weight(group_list[k](i));
                }

//                double weight_sum = test_weight.sum();
//
//                for (i = 0; i < group_list[k].size(); i++) {
//                    test_weight(i) = test_weight(i) / weight_sum;
//                }

//                std::cout<<"cv_3"<<endl;

                algorithm->update_train_mask(train_mask);
                algorithm->fit();

//                std::cout<<"cv_4"<<endl;

                Eigen::VectorXd coef(test_n);
                Eigen::VectorXd one = Eigen::VectorXd::Ones(test_n);

                for(i=0;i<=test_n-1;i++) {
                    coef(i) = algorithm->get_coef0();
                }
                Eigen::VectorXd xbeta_exp = test_x*algorithm->get_beta()+coef;
                for(i=0;i<=test_n-1;i++) {
                    if(xbeta_exp(i)>25.0) xbeta_exp(i) = 25.0;
                    if(xbeta_exp(i)<-25.0) xbeta_exp(i) = -25.0;
                }
                xbeta_exp = xbeta_exp.array().exp();
                Eigen::VectorXd pr = xbeta_exp.array()/(xbeta_exp+one).array();

                loss_list(k) = -2*(test_weight.array()*((test_y.array()*pr.array().log())+(one-test_y).array()*(one-pr).array().log())).sum();
            }
//            std::cout<<"cv"<<endl;
            return loss_list.sum() / loss_list.size();

        }


    }

    double ic(Algorithm* algorithm, Data &data){
        if(this->is_cv){
            return this->test_loss(algorithm, data);
        }
        else
        {
            if(ic_type == 1){
                return this->train_loss(algorithm, data) + 2.0 * algorithm->get_sparsity_level();
            } else if (ic_type == 2){
                return this->train_loss(algorithm, data) + log(double(data.get_n())) * algorithm->get_sparsity_level();
            } else if (ic_type == 3){
                return this->train_loss(algorithm, data) + log(double(data.get_p())) * log(log(double(data.get_n()))) * algorithm->get_sparsity_level();
            } else if (ic_type == 4){
                return this->train_loss(algorithm, data) + (log(double(data.get_n())) + 2 * log(double(data.get_p()))) * algorithm->get_sparsity_level();
            }
        }
    }

};

class PoissonMetric : public Metric {
public:

    PoissonMetric(int ic_type, bool is_cv, int K=0):Metric(ic_type, is_cv, K){};

    double train_loss(Algorithm* algorithm, Data &data){
        int i;
        int n = data.get_n();
        Eigen::VectorXd coef(n);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);

        for(i=0;i<=n-1;i++) {
            coef(i) = algorithm->get_coef0();
        }
        Eigen::VectorXd xbeta = data.x*algorithm->get_beta()+coef;
        for(int i=0;i<=n-1;i++) {
            if(xbeta(i)>30.0) xbeta(i) = 30.0;
            if(xbeta(i)<-30.0) xbeta(i) = -30.0;
        }
        Eigen::VectorXd xbeta_exp = xbeta.array().exp();

        return (data.y - xbeta_exp).squaredNorm()/double(data.get_n());
    }

    double test_loss(Algorithm *algorithm, Data &data) {
        if (!is_cv) {
            return this->train_loss(algorithm, data);
        } else {
//            std::cout<<"cv"<<endl;
            int k, i, kk, kkk;
            int n = data.get_n();
            int p = data.get_p();
            int temp, change_index;
            Eigen::VectorXi index_list(n);
            Eigen::VectorXd loss_list(this->K);

            for (i = 0; i < n; i++) {
                index_list(i) = int(i);
            }
            for (i = n - 1; i > 0; i--) {
                change_index = rand() % int(i + 1);
                temp = index_list[i];
                index_list[i] = index_list[change_index];
                index_list[change_index] = temp;
            }

            vector<Eigen::VectorXi> group_list(this->K);
            int group_size = int(n / this->K);
            for (k = 0; k < (this->K - 1); k++) {
                group_list[k] = index_list.segment(int(k * group_size), group_size);
            }

            group_list[this->K - 1] = index_list.segment(int((this->K - 1) * group_size),
                                                         n - (this->K - 1) * group_size);
            for (k = 0; k < this->K; k++) {
                int train_x_size = 0;
                for (kk = 0; kk < this->K; kk++) {
                    if (kk != k) {
                        train_x_size = train_x_size + group_list[kk].size();
                    }
                }

                // get train_mask
                Eigen::VectorXi train_mask(train_x_size);
                i = 0;
                for (kk = 0; kk < this->K; kk++) {
                    if (kk != k) {
                        for (kkk = 0; kkk < group_list[kk].size(); kkk++) {
                            train_mask[i] = group_list[kk](kkk);
                            i++;
                        }
                    }
                }

                //get test_x, test_y
                int test_n = group_list[k].size();

//                std::cout<<"test size"<<test_n<<endl;
                Eigen::MatrixXd test_x(test_n, p);
                Eigen::VectorXd test_y(test_n);
                Eigen::VectorXd test_weight(test_n);

                for (i = 0; i < group_list[k].size(); i++) {
                    test_x.row(i) = data.x.row(group_list[k](i));
                    test_y(i) = data.y(group_list[k](i));
                    test_weight(i) = data.weight(group_list[k](i));
                }

                algorithm->update_train_mask(train_mask);
                algorithm->fit();

                Eigen::VectorXd coef(test_n);
                Eigen::VectorXd one = Eigen::VectorXd::Ones(test_n);

                for(i=0;i<=test_n-1;i++) {
                    coef(i) = algorithm->get_coef0();
                }
                Eigen::VectorXd xbeta_exp = test_x*algorithm->get_beta()+coef;
                for(i=0;i<=test_n-1;i++) {
                    if(xbeta_exp(i)>25.0) xbeta_exp(i) = 25.0;
                    if(xbeta_exp(i)<-25.0) xbeta_exp(i) = -25.0;
                }
                xbeta_exp = xbeta_exp.array().exp();
                loss_list(k) = (test_y - xbeta_exp).squaredNorm()/double(test_n);
            }
//            std::cout<<"cv"<<endl;
            return loss_list.sum() / loss_list.size();

        }


    }

    double ic(Algorithm* algorithm, Data &data){
        if(this->is_cv){
            return this->test_loss(algorithm, data);
        }
        else
        {
            if(ic_type == 1){
                return double(data.get_n()) * this->train_loss(algorithm, data) + 2.0 * algorithm->get_sparsity_level();
            } else if (ic_type == 2){
                return double(data.get_n()) * this->train_loss(algorithm, data) + log(double(data.get_n())) * algorithm->get_sparsity_level();
            } else if (ic_type == 3){
                return double(data.get_n()) * this->train_loss(algorithm, data) + log(double(data.get_p())) * log(log(double(data.get_n()))) * algorithm->get_sparsity_level();
            }else if (ic_type == 3){
                return double(data.get_n()) * this->train_loss(algorithm, data) + (log(double(data.get_n())) + 2 * log(double(data.get_p()))) * algorithm->get_sparsity_level();
            }
        }
    }

};

class CoxMetric : public Metric {
public:

    CoxMetric(int ic_type, bool is_cv, int K=0):Metric(ic_type, is_cv, K){};

    double train_loss(Algorithm* algorithm, Data &data){
        return -2 * loglik_cox(data.x, data.y, algorithm->get_beta(), data.weight);
    };

    double test_loss(Algorithm* algorithm, Data &data){
        if(!this->is_cv){
            return -2 * loglik_cox(data.x, data.y, algorithm->get_beta(), data.weight);
        }
        else{
            int k, i, kk, kkk;
            int n = data.get_n();
            int p = data.get_p();
            int temp, change_index;
            Eigen::VectorXi index_list(n);


            Eigen::VectorXd loss_list(this->K);

            for(i=0;i<n;i++)
            {
                index_list(i) = int(i);
            }
            for (i=n-1;i>0;i--)
            {
                change_index = rand() % int(i+1);
                temp = index_list[i];
                index_list[i] = index_list[change_index];
                index_list[change_index] = temp;
            }

            vector<Eigen::VectorXi> group_list(this->K);
            int group_size = int(n / this->K);
            for(k=0;k<(this->K - 1);k++)
            {
                group_list[k] = index_list.segment(int(k * group_size), group_size);
            }




            group_list[this->K - 1] = index_list.segment(int((this->K - 1) * group_size), n - int(int(this->K - 1) * group_size));

//            for(k=0;k<this->K;k++)
//            {
//                for(kk=0;kk<group_list[k].size();kk++)
//                    std::cout<<group_list[k][kk]<<" ";
//                std::cout<<endl;
//            }


            for(k=0;k<this->K;k++)
            {
                int train_x_size = 0;
                for(kk=0;kk<this->K;kk++)
                {
                    if(kk != k) {
                        train_x_size = train_x_size + group_list[kk].size();
                    }
                }

                // get train_mask

                Eigen::VectorXi train_mask(train_x_size);
                i = 0;
                for(kk=0;kk<this->K;kk++)
                {
                    if(kk != k)
                    {
                        for(kkk=0;kkk<group_list[kk].size();kkk++)
                        {
                            train_mask(i) = group_list[kk](kkk);
                            i++;
                        }
                    }
                }
//                std::cout<<"i = "<<i<<endl;

//                for(i=0;i<train_x_size;i++)
//                {
//                    std::cout<<train_mask(i)<<" ";
//                }

                //get test_x, test_y
                Eigen::MatrixXd test_x(group_list[k].size(), p);
                Eigen::VectorXd test_y(group_list[k].size());
                Eigen::VectorXd test_weight(group_list[k].size());

//                std::cout<<"train size"<< train_x_size<<endl;
//                std::cout<<"test size"<< group_list[k].size()<<endl;

                for(i=0;i<group_list[k].size();i++){
                    test_x.row(i) = data.x.row(group_list[k](i));
                    test_y(i) = data.y(group_list[k](i));
                    test_weight(i) = data.weight(group_list[k](i));
                };

                algorithm->update_train_mask(train_mask);
                algorithm->fit();

                loss_list(k) = -2 * loglik_cox(test_x, test_y, algorithm->get_beta(), test_weight);
            }
//            for(i=0;i<loss_list.size();i++)
//                std::cout<<loss_list(i)<<" "<<endl;
            return loss_list.sum() / double(loss_list.size());
//            std::cout<<"cv_end"<<endl;
        }

    };

    double ic(Algorithm* algorithm, Data &data){
        if(this->is_cv){
            return this->test_loss(algorithm, data);
        }
        else
        {
            if(ic_type == 1){
                return double(data.get_n()) * log(this->train_loss(algorithm, data)) + 2.0 * algorithm->get_sparsity_level();
            } else if (ic_type == 2){
                return double(data.get_n()) * log(this->train_loss(algorithm, data)) + log(double(data.get_n())) * algorithm->get_sparsity_level();
            } else if (ic_type == 3){
                return double(data.get_n()) * log(this->train_loss(algorithm, data)) + log(double(data.get_p())) * log(log(double(data.get_n()))) * algorithm->get_sparsity_level();
            } else if (ic_type == 4){
                return double(data.get_n()) * log(this->train_loss(algorithm, data)) + (log(double(data.get_n())) + 2 * log(double(data.get_p()))) * algorithm->get_sparsity_level();
            }

        }
    };

};

#endif //SRC_METRICS_H
