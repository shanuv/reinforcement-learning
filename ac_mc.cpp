/*Implementation of the Actor critic algorithm for the Mountaincar domain using the Fourier basis


To compile use the C++11 compiler and provide the path to the Eigen library- 

g++ -std=c++11 -I ~/Documents/eigen ac_mc.cpp -o ac_mc.o

To run the same - 

./ac_mc.o <alpha> <lambda>

Typical values are - 0.001 0.9

The code implements Softmax policy

*/
#include <stdio.h>
#include <iostream>
#include <random>
#include <math.h>
#include <Eigen/Dense>
#include <ctime>

using namespace std;
using namespace Eigen;

#define N 100						//# Episodes
#define PI 3.14159265
#define trials 100						//# Trials to compute the expected return
#define basis 3
#define dim 2

typedef struct Mountaincarstate{
	double x;
	double v;
} State;

int main(int argc,char* argv[]){

	double alpha = atof(argv[1]),gamma=1,old_x,old_v,error[trials][N]={0},delta;
	State initial_state={-0.5,0},final_state;
	double new_x,new_v,lambda=atof(argv[2]);
	
	double value_s,value_sdash;
	MatrixXd cmatrix((int)pow(basis+1,dim),2);
	Vector2d dummy;
	VectorXd phis((int)pow(basis+1,dim)),phisdash((int)pow(basis+1,dim)),eligibility_v((int)pow(basis+1,dim)),weights((int)pow(basis+1,dim));		
	MatrixXd eligibility_theta(3,(int)pow(basis+1,dim)),theta(3,(int)pow(basis+1,dim));
	
	int counter[2]={0},index;
	for(int i=0;i<(int)pow(basis+1,dim);i++){
		phis(i)=0;
		phisdash(i)=0;
		counter[1]=i%(basis+1);
		counter[0]=i/(basis+1);
		for(int j=0;j<2;j++)
			cmatrix(i,j) = counter[j];
 		
	}
//	cout << "I am in line "<< __LINE__ << endl;
	mt19937 gen(time(NULL));
	double weights_softmax[3];
	int action,flag=0,trial=0;
	
	while(trial<trials){
		for(int j=0;j<3;j++)
			for(int i=0;i<(int)pow(basis+1,dim);i++){
				weights(i)=0;
				theta(j,i)=0;
		}

		for( int count =0; count < N; count++){
			int result=0;								//Beginning of an episode
			State initial_state={-0.5,0.0};				//s0
			old_x = initial_state.x;
			old_v = initial_state.v;
			flag = 0;									//Flag keeping track of terminal state
			new_x=0;new_v=0;
			for(int j=0;j<3;j++)
				for(int i=0;i<(int)pow(basis+1,dim);i++){
					eligibility_v(i)=0;
					eligibility_theta(j,i)=0;
			}
			double sum=0;	
//			cout << "I am in line "<< __LINE__ << endl;	
			while(new_x!=0.5){

				dummy(0)=old_x; dummy(0) = (dummy(0) + 1.2) /(1.7);
				dummy(1)=old_v; dummy(1) = (dummy(1) + 0.07) / (2*0.07);
				phis=PI*cmatrix*dummy;
//cout << "I am in line "<< __LINE__ << endl;
				for(int i=0;i<(int)pow(basis+1,dim);i++)
					phis(i)=cos(phis(i));
				value_s = weights.transpose()*phis;
				sum=0;
				for (int counter=0;counter<3;counter++){
					double expression=0;
					for(int i=0;i<(int)pow(basis+1,dim);i++)
						expression+=theta(counter,i)*phis(i);
					weights_softmax[counter]=exp(expression);
					sum+=weights_softmax[counter];
				}
				for (int counter=0;counter<3;counter++)
					weights_softmax[counter]/=sum;
				discrete_distribution<> distr(weights_softmax,  weights_softmax + sizeof weights_softmax/sizeof weights_softmax[0]);
				action = distr(gen);
//cout << "I am in line "<< __LINE__ << endl;
				new_v = old_v + 0.001*(action-1)-0.0025*cos(3*old_x);			//Dynamics
				new_x = old_x + new_v;
				
				if(new_x<-1.2){
					new_x=-1.2;new_v=0;
				}
				if(new_x>0.5){
					new_x=0.5;new_v=0;
				}
				dummy(0)=new_x; dummy(0) = (dummy(0) + 1.2) /(1.7);
				dummy(1)=new_v; dummy(1) = (dummy(1) + 0.07) / (2*0.07);
				if (dummy(0)==1)
					flag=1;
				if(dummy(1)>1){
					dummy(1)=1;new_v=0.07;
				}
				if(dummy(1)<0){
					dummy(1)=0;new_v=-0.07;
				}
				if(!flag){
					phisdash=PI*cmatrix*dummy;
					for(int i=0;i<(int)pow(basis+1,dim);i++)
						phisdash(i)=cos(phisdash(i));
					value_sdash = weights.transpose()*phisdash;		//q(s',a')
				}
				else
					value_sdash = 0;
//cout << "I am in line "<< __LINE__ << endl;					
				for(int i=0;i<(int)pow(basis+1,dim);i++)
					eligibility_v(i)=gamma*lambda*eligibility_v(i)+phis(i);
				
				delta = -1+gamma*value_sdash-value_s;

				for(int i=0;i<(int)pow(basis+1,dim);i++)
					weights(i)+=alpha*delta*eligibility_v(i);

				for(int j=0;j<3;j++)
					for(int i=0;i<(int)pow(basis+1,dim);i++)
						eligibility_theta(j,i)=gamma*lambda*eligibility_theta(j,i)-weights_softmax[j]*phis(i);

				for(int i=0;i<(int)pow(basis+1,dim);i++) 
					eligibility_theta(action,i)+=phis(i);

				for(int j=0;j<3;j++)
					for(int i=0;i<(int)pow(basis+1,dim);i++)
						theta(j,i)+=alpha*delta*eligibility_theta(j,i);
//cout << "I am in line "<< __LINE__ << endl;
				if(flag)
					break;
				result+= -1;				//Rt = -1
				final_state={new_x,new_v};
//cout << "I am in line "<< __LINE__ << endl;				
				old_x=new_x;				//s=s'
				old_v=new_v;
				if(result<-1000)
					break;
				
		}
		error[trial][count]=result;
	}
	trial++;
	
}

	double sum=0.0,sum_error,mean[N],std_dev[N];			//Calculating statistics
	for(int i=0;i<N;i++){
		sum=0.0,sum_error=0.0;
		for(int j=0;j<trials;j++)
			sum+=error[j][i];
			
		
		mean[i] = sum/trials;

		for(int j=0;j<trials;j++)			
			sum_error+=pow(error[j][i]-mean[i],2);
					
		sum_error/=(trials-1);
		
		std_dev[i] = sqrt(sum_error);
	}
	for(int i =0;i<N;i++)
		cout<< mean[i] << " " << i << " " << std_dev[i]<< endl;
	
	return 0;
}
