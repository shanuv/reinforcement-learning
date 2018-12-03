/*Implementation of the Sarsa-lambda algorithm for the Mountaincar domain using the Fourier basis


To compile use the C++11 compiler and provide the path to the Eigen library- 

g++ -std=c++11 -I ~/Documents/eigen sarsalambda_mountaincar.cpp -o sarsalambda_mountaincar.o

To run the same - 

./sarsalambda_mountaincar.o <alpha> <epsilon> <lambda>

Typical values are - 0.01 0.01 0.9

The code implements epsilon-greedy policy

*/
#include <stdio.h>
#include <iostream>
#include <random>
#include <math.h>
#include <Eigen/Dense>
#include <ctime>

using namespace std;
using namespace Eigen;

#define N 100							//# Episodes
#define PI 3.14159265
#define trials 100						//# Trials to compute the expected return
#define basis 7
#define dim 2

typedef struct Mountaincarstate{
	double x;
	double v;
} State;

int main(int argc,char* argv[]){

	double alpha = atof(argv[1]),gamma=1,epsilon=atof(argv[2]),old_x,old_v,error[trials][N]={0};
	State initial_state={-0.5,0},final_state;
	double new_x,new_v,lambda=atof(argv[3]);
	
	Vector3d value_qsa,value_qsdashadash;
	MatrixXd cmatrix((int)pow(basis+1,dim),2);
	Vector2d dummy;
	VectorXd phisa((int)pow(basis+1,dim)),phisdashadash((int)pow(basis+1,dim));		
	MatrixXd weights((int)pow(basis+1,dim),3),eligibility((int)pow(basis+1,dim),3);
	
	int counter[2]={0},index;
	for(int i=0;i<(int)pow(basis+1,dim);i++){
		phisa(i)=0;
		phisdashadash(i)=0;
		counter[1]=i%(basis+1);
		counter[0]=i/(basis+1);
		for(int j=0;j<2;j++)
			cmatrix(i,j) = counter[j];
 		
	}
	
	mt19937 gen(time(NULL));
	double weights_epsilongreedy[] ={epsilon/3,epsilon/3,epsilon/3}; 
	discrete_distribution<> distr(weights_epsilongreedy,weights_epsilongreedy + sizeof weights_epsilongreedy/sizeof weights_epsilongreedy[0]);
	int action,actiondash,flag=0,trial=0;
	
	while(trial<trials){
		
		for(int i=0;i<(int)pow(basis+1,dim);i++)
			for(int j=0;j<3;j++){
				weights(i,j)=0;
		}

		for( int count =0; count < N; count++){
			int result=0;								//Beginning of an episode
			State initial_state={-0.5,0.0};				//s0
			old_x = initial_state.x;
			old_v = initial_state.v;
			flag = 0;									//Flag keeping track of terminal state
			new_x=0;new_v=0;
			for(int i=0;i<(int)pow(basis+1,dim);i++)
				for(int j=0;j<3;j++)
					eligibility(i,j)=0;
			
			dummy(0)=old_x; dummy(0) = (dummy(0) + 1.2) /(1.7);
			dummy(1)=old_v; dummy(1) = (dummy(1) + 0.07) / (2*0.07);
			phisa=PI*cmatrix*dummy;

			for(int i=0;i<(int)pow(basis+1,dim);i++)
				phisa(i)=cos(phisa(i));

			value_qsa = weights.transpose()*phisa;		//
			double max = value_qsa(0);index=0;
			for(int i=1;i<3;i++){
				if(value_qsa(i)>max){
					index = i;
					max = value_qsa(i);
				}
			}
			weights_epsilongreedy[index]+=(1-epsilon);
			discrete_distribution<> distr(weights_epsilongreedy,weights_epsilongreedy + sizeof weights_epsilongreedy/sizeof weights_epsilongreedy[0]);
			action = distr(gen);
			weights_epsilongreedy[index]-=(1-epsilon);

			while(new_x!=0.5){
				weights_epsilongreedy[0]=epsilon/3;
				weights_epsilongreedy[1]=epsilon/3;
				weights_epsilongreedy[2]=epsilon/3;

				new_v = old_v + 0.001*(action-1)-0.0025*cos(3*old_x);			//Dynamics
				new_x = old_x + new_v;
				//cout << new_x << endl;
				dummy(0)=old_x; dummy(0) = (dummy(0) + 1.2) /(1.7);
				dummy(1)=old_v; dummy(1) = (dummy(1) + 0.07) / (2*0.07);
				phisa=PI*cmatrix*dummy;

				for(int i=0;i<(int)pow(basis+1,dim);i++)
					phisa(i)=cos(phisa(i));			

				value_qsa = weights.transpose()*phisa;						//q(s,a)
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
					phisdashadash=PI*cmatrix*dummy;
					for(int i=0;i<(int)pow(basis+1,dim);i++)
						phisdashadash(i)=cos(phisdashadash(i));
					value_qsdashadash = weights.transpose()*phisdashadash;		//q(s',a')
				}
				else{ 
					value_qsdashadash(0) = 0;
					value_qsdashadash(1) = 0;
					value_qsdashadash(2) = 0;
				}
				max = value_qsdashadash(0);index=0;
				for(int i=1;i<3;i++){
					if(value_qsdashadash(i)>max){
						index = i;
						max = value_qsdashadash(i);
					}
				}
				weights_epsilongreedy[index]+=(1-epsilon);
				discrete_distribution<> distr(weights_epsilongreedy,weights_epsilongreedy + sizeof weights_epsilongreedy/sizeof weights_epsilongreedy[0]);
				actiondash = distr(gen);						//Choosing a'
				weights_epsilongreedy[index]-=(1-epsilon);	
				for (int j=0;j<(int)pow(basis+1,dim);j++)
					for(int k=0;k<3;k++)
						eligibility(j,k) = gamma*lambda*eligibility(j,k);
				for (int j=0;j<(int)pow(basis+1,dim);j++)
					eligibility(j,action)+=phisa(j);
				for (int j=0;j<(int)pow(basis+1,dim);j++)
					for(int k=0;k<3;k++)
						weights(j,k) = weights(j,k)+(alpha*(-1+gamma*value_qsdashadash(actiondash)-value_qsa(action))*eligibility(j,k)); //Sarsa update
				
				if(flag)
					break;
				result+= -1;				//Rt = -1
				final_state={new_x,new_v};
				action = actiondash;		//a=a'
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
