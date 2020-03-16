#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
#else

#include <Eigen/Eigen>
#include "List.h"

#endif

#include <iostream>
#include <typeinfo>
#include <vector>

using namespace std;
//using namespace Eigen;

//template<typename T> void List::add(string name, T value)
//{
//	double temp_double;
//	Eigen::MatrixXd temp_MatrixXd;
//	Eigen::VectorXd temp_VectorXd;
//	cout<<typeid(temp_double).name()<<endl;
//	cout<<typeid(temp_MatrixXd).name()<<endl;
//	cout<<typeid(temp_VectorXd).name()<<endl;
////	cout<<typeid(typeid(value).name()).name()<<endl;
////	bool a;
////	a = typeid(value).name() == "d";
////	cout<<a<<endl;
//	if(typeid(value).name() == typeid(temp_double).name())
//	{
//		cout<<"value in double add"<<endl;
//		vector_double.push_back(value);
//		vector_double_name.push_back(name);
//	}
////	else if(typeid(value).name() == typeid(temp_MatrixXd).name())
////	{
////		cout<<"value in MatrixXd add"<<endl;
////		vector_MatrixXd.push_back(value);
////		vector_MatrixXd_name.push_back(name);	
////	}
////	else if(typeid(value).name() == typeid(temp_VectorXd).name())
////	{
////		cout<<"value in VectorXd add"<<endl;
////		vector_VectorXd.push_back(value);
////		vector_VectorXd_name.push_back(name);	
////	}
//}
//
//template<typename T> T List::get_value_by_name(string name, T type)
//{
//	double temp_double;
//	Eigen::MatrixXd temp_MatrixXd;
//	Eigen::VectorXd temp_VectorXd;
//	T value;
//	int i;
//	if(typeid(value).name() == typeid(temp_double).name())
//	{
//		for(i=0;i<vector_double_name.size();i++)
//		{
//			cout<<"value in get double"<<endl;
//			if(vector_double_name[i] == name)
//			{
//				return vector_double[i];
//			}
//		}
//	}
//	else if(typeid(value).name() == typeid(temp_MatrixXd).name())
//	{
//		for(i=0;i<vector_MatrixXd_name.size();i++)
//		{
//			cout<<"value in get MatrixXd"<<endl;
//			if(vector_MatrixXd_name[i] == name)
//			{
//				return vector_MatrixXd[i];
//			}
//		}
//	}
//	else if(typeid(value).name() == typeid(temp_VectorXd).name())
//	{
//		for(i=0;i<vector_VectorXd_name.size();i++)
//		{
//			cout<<"value in get VectorXd"<<endl;
//			if(vector_VectorXd_name[i] == name)
//			{
//				return vector_VectorXd[i];
//			}
//		}
//	}
//}

void List::add(string name, int value) {
//	cout<<"value in int add"<<endl;
    vector_int.push_back(value);
    vector_int_name.push_back(name);
}

void List::add(string name, double value) {
//	cout<<"value in double add"<<endl;
    vector_double.push_back(value);
    vector_double_name.push_back(name);
}

void List::add(string name, MatrixXd value) {
//	cout<<"value in MatrixXd add"<<endl;
    vector_MatrixXd.push_back(value);
    vector_MatrixXd_name.push_back(name);
}

void List::add(string name, VectorXd value) {
//	cout<<"value in VectorXd add"<<endl;
    vector_VectorXd.push_back(value);
    vector_VectorXd_name.push_back(name);
}

void List::add(string name, VectorXi value) {
//	cout<<"value in VectorXi add"<<endl;
    vector_VectorXi.push_back(value);
    vector_VectorXi_name.push_back(name);
}

void List::get_value_by_name(string name, int &value) {
    std::size_t i;
    for (i = 0; i < vector_int_name.size(); i++) {
//		cout<<"value in get double"<<endl;
        if (vector_int_name[i] == name) {
            value = vector_int[i];
            break;
        }
    }
}

void List::get_value_by_name(string name, double &value) {
    std::size_t i;
    for (i = 0; i < vector_double_name.size(); i++) {
//		cout<<"value in get double"<<endl;
        if (vector_double_name[i] == name) {
            value = vector_double[i];
            break;
        }
    }
}

void List::get_value_by_name(string name, MatrixXd& value) {
    std::size_t i;
//	int j;
    for (i = 0; i < vector_MatrixXd_name.size(); i++) {
//		cout<<"value in get MatrixXd"<<endl;
//		cout<<vector_MatrixXd_name[i]<<endl;
//		cout<<name<<endl;
//		cout<<(vector_MatrixXd_name[i] == name)<<endl;
        if (vector_MatrixXd_name[i] == name) {
            value = vector_MatrixXd[i];
            break;
        }
    }
}

void List::get_value_by_name(string name, VectorXd& value) {
    std::size_t i;
    for (i = 0; i < vector_VectorXd_name.size(); i++) {
//		cout<<"value in get VectorXd"<<endl;
        if (vector_VectorXd_name[i] == name) {
            value = vector_VectorXd[i];
            break;
        }
    }
}

void List::get_value_by_name(string name, VectorXi& value) {
    std::size_t i;
    for (i = 0; i < vector_VectorXi_name.size(); i++) {
//		cout<<"value in get VectorXi"<<endl;
        if (vector_VectorXi_name[i] == name) {
            value = vector_VectorXi[i];
            break;
        }
    }
}
