#include <stdio.h>
#include <iostream>
#include <random>
#include <math.h>
#include <ctime>
using namespace std;

#define episodes 150
#define trials 200
#define gridworld_size 5

int return_state(int i,int j){
	int array[5][5]={{1,2,3,4,5},{6,7,8,9,10},{11,12,-1,13,14},{15,16,-1,17,18},{19,20,21,22,23}};
	return array[i][j];
}

int main(int argc, char* argv[]){

	int grid[gridworld_size][gridworld_size]={0};
	double statevalue[23]={0},error[trials][episodes],eligibility_theta[23][4],theta[23][4]={0},eligibility_v[23];
	double gamma=0.9,alpha = atof(argv[1]),lambda = atof(argv[2]),delta;
	int action,action_taken,curr_state,calc_state;

	grid[4][4]=10;
	grid[4][2]=-10;

	random_device rand_gen;
																		//Action probability
	mt19937 gen(time(NULL));
	double weights[] = {0.8, 0.05, 0.05,0.1};
	discrete_distribution<> d(weights,  weights + sizeof weights/sizeof weights[0]);

	mt19937 generator(rand_gen());
	double weights_softmax[4];  //Calculate pi on the fly
	/*Action key - 0 --> AU - U | R | L | S
				 - 1 --> AD - D | L | R | S
				 - 2 --> AL - L | U | D | S
				 - 3 --> AR - R | D | U | S
	*/
	//double reward[episodes] = {0.0};
	int trial = 0;
	while(trial<trials){
		for(int i=0;i<23;i++){
			statevalue[i]=1;
			for(int j=0;j<4;j++)
				theta[i][j]=0;
		}

		for( int count =0; count < episodes; count++){
			clock_t start=clock();
			int curr_i=0, curr_j=0, calc_i=0, calc_j=0;
			int time = 0;
			double result=0.0,sum=0;
			for(int i =0;i<23;i++){
				eligibility_v[i]=0;
				for(int j=0;j<4;j++)
					eligibility_theta[i][j]=0;
			}
			while(!(curr_i == 4 && curr_j == 4)){

				curr_state = return_state(curr_i,curr_j)-1;
				sum=0;
				for (int counter=0;counter<4;counter++){
					weights_softmax[counter]=exp(theta[curr_state][counter]*statevalue[curr_state]);
					sum+=weights_softmax[counter];
				}
				for (int counter=0;counter<4;counter++)
					weights_softmax[counter]/=sum;
				discrete_distribution<> distr(weights_softmax,  weights_softmax + sizeof weights_softmax/sizeof weights_softmax[0]);
				action = distr(generator);
				action_taken = d(gen);

				if ((action == 0 && action_taken == 0 )||(action == 2 && action_taken == 1)||(action == 3 && action_taken == 2) ){
						 if(curr_i == 0 )
								calc_i = curr_i;
							else
								calc_i = curr_i - 1;
							calc_j = curr_j;
					}
				else if ((action == 1 && action_taken == 0 )||(action == 2 && action_taken == 2)||(action == 3 && action_taken == 1) ){
						 if(curr_i == 4)
								calc_i = curr_i;
							else
								calc_i = curr_i + 1;
							calc_j = curr_j;
					}
				else if((action == 2 && action_taken == 0 )||(action == 1 && action_taken == 1)||(action == 0 && action_taken == 2) ){
						 if(curr_j == 0)
								calc_j = curr_j;
							else
								calc_j = calc_j - 1;
							calc_i = curr_i;
						}
				else if ((action == 3 && action_taken == 0 )||(action == 1 && action_taken == 2)||(action == 0 && action_taken == 1) ){
						 if(curr_j == 4)
								calc_j = curr_j;
							else
								calc_j = calc_j + 1;
							calc_i = curr_i;
						}
				else if(action_taken == 3){
						calc_i = curr_i;
						calc_j = curr_j;
				}

				if(calc_i == 2 && calc_j == 2){		//Obstacle scenario
					calc_i = curr_i;
					calc_j = curr_j;
				}

				if(calc_i == 3 && calc_j == 2){		//Obstacle scenario
					calc_i = curr_i;
					calc_j = curr_j;
				}

				result+=grid[calc_i][calc_j]*pow(gamma, time);
				calc_state = return_state(calc_i,calc_j)-1;
			
				for(int i=0;i<23;i++)
					eligibility_v[i]=gamma*lambda*eligibility_v[i];
				eligibility_v[curr_state]+=1;

				delta = grid[calc_i][calc_j]+gamma*statevalue[calc_state]-statevalue[curr_state];

				for(int i=0;i<23;i++)
					statevalue[i]+=alpha*delta*eligibility_v[i];
				for(int i=0;i<23;i++)
					for(int j=0;j<4;j++)
						eligibility_theta[i][j]=gamma*lambda*eligibility_theta[i][j];
				for(int j=0;j<4;j++)
					eligibility_theta[curr_state][j]+=-1*weights_softmax[j]*statevalue[curr_state];
				eligibility_theta[curr_state][action]+=statevalue[curr_state];
				for(int i=0;i<23;i++)
					for(int j=0;j<4;j++)
						theta[i][j]+=alpha*delta*eligibility_theta[i][j];

				curr_i = calc_i;					//New State
				curr_j = calc_j;

				time++;
				if(time>200)
					break;
				//epsilon = 1/time;
			/*cout << count <<"Done"<< result << endl;	*/
			}
			/*clock_t end=clock();
			cout << "Time" << double(end-start)/CLOCKS_PER_SEC;*/
			//cout << count << " "<< time << endl;
			//reward[count]+= result;
			error[trial][count] = result;
			/*for(int j=0;j<4;j++){
				for(int i=0;i<23;i++)
					cout << actionvalue[i][j]<<" ";
				cout << endl;
			}*/
			//cout << "End of episode "<<count << "in a time of "<<time<<endl;
		}
		trial++;
		//cout << "End of trial"<<endl;
}

/*for(int i=0;i<trials;i++){
	for(int j=0;j<episodes;j++)
		cout<< error[i][j]<<" ";
	cout << endl;
}
*/
double sum=0.0,sum_error,mean[episodes],std_dev[episodes];
	for(int i=0;i<episodes;i++){
		sum=0.0,sum_error=0.0;
		for(int j=0;j<trials;j++)
			sum+=error[j][i];
			//cout << error[i][j]<< endl;
		
		mean[i] = sum/trials;

		for(int j=0;j<trials;j++)			
			sum_error+=pow(error[j][i]-mean[i],2);
					
		sum_error/=(trials-1);
		//cout << sum_error << endl;
		std_dev[i] = sqrt(sum_error);
	}
	for(int i =0;i<episodes;i++)
		cout<< mean[i] << " " << i << " " << std_dev[i]<< endl;
	
	
	return 0;
}
