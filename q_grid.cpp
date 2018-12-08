/* Code implementing Part 2 of Assignment 1.

Compilation Instructions -

On a command-line terminal, compile the code using the C++11 compiler with the following command.

$ g++ -std=c++0x gridworld.cpp

To run the generated executable,

$ ./a.out

*/

#include <stdio.h>
#include <iostream>
#include <random>
#include <math.h>

using namespace std;
#define episodes 100
#define trials 100
#define gridworld_size 5

int return_state(int i,int j){
	int array[5][5]={{1,2,3,4,5},{6,7,8,9,10},{11,12,-1,13,14},{15,16,-1,17,18},{19,20,21,22,23}};
	return array[i][j];
}

int main(int argc, char* argv[]){

	int grid[gridworld_size][gridworld_size]={0};
	double actionvalue[23][4]={0},error[trials][episodes],eligibility[23][4];
	double gamma=0.9,alpha = atof(argv[1]),epsilon=atof(argv[2]),lambda = atof(argv[3]);
	int action,action_taken,curr_state,calc_state,max,max_index,calc_action,calc_action_taken;

	grid[4][4]=10;
	grid[4][2]=-10;

	random_device rand_gen;
																		//Action probability
	mt19937 gen(time(NULL));
	//double weights[]={0.238061,0.244509,0.263692,0.30419};
	double weights[] = {0.8, 0.05, 0.05,0.1};
	discrete_distribution<> d(weights,  weights + sizeof weights/sizeof weights[0]);

	//For generating random numbers to denote the action
	const int start=0, end=3;

	mt19937 generator(rand_gen());
	double weights_epsilongreedy[] ={epsilon/4,epsilon/4,epsilon/4,epsilon/4}; //Will have to be reordered based on the optimal action for the step
	/*Action key - 0 --> AU - U | R | L | S
				 - 1 --> AD - D | L | R | S
				 - 2 --> AL - L | U | D | S
				 - 3 --> AR - R | D | U | S
	*/
	//double reward[episodes] = {0.0};
	int trial = 0;
	while(trial<trials){
		for(int i=0;i<23;i++)
			for(int j=0;j<4;j++)
				actionvalue[i][j]=0;
		for( int count =0; count < episodes; count++){

			int curr_i=0, curr_j=0, calc_i=0, calc_j=0;
			int time = 0;
			double result=0.0;
			for(int i =0;i<23;i++)
				for(int j=0;j<4;j++)
					eligibility[i][j]=0;

		
		while(!(curr_i == 4 && curr_j == 4)){

			curr_state = return_state(curr_i,curr_j);
			max = actionvalue[curr_state-1][0];
			for (int counter=1;counter<4;counter++){
				if (actionvalue[curr_state-1][counter] > max){
					max = actionvalue[curr_state-1][counter];
					max_index = counter;
				}
			}
			for (int i =0;i<4;i++)
			weights_epsilongreedy[i]=epsilon/4;
			
			weights_epsilongreedy[max_index]+=(1-epsilon);
			discrete_distribution<> distr(weights_epsilongreedy, weights_epsilongreedy + sizeof weights_epsilongreedy/sizeof weights_epsilongreedy[0]);
			action = distr(generator);
			weights_epsilongreedy[max_index]-=(1-epsilon);

			action_taken = d(gen);

									//Action succeeds/ veer left/veer right/stay

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
			calc_state = return_state(calc_i,calc_j);

			max = actionvalue[calc_state-1][0];
			for (int counter=1;counter<4;counter++){
				if (actionvalue[calc_state-1][counter] > max){
					max = actionvalue[calc_state-1][counter];
					max_index = counter;
				}
			}
			for(int i=0;i<23;i++)
					for (int j=0;j<4;j++)
						eligibility[i][j]=gamma*lambda*eligibility[i][j];
			eligibility[curr_state-1][action]+=1;
			for(int i=0;i<23;i++)
					for (int j=0;j<4;j++)
						actionvalue[curr_state-1][action]+=alpha*(grid[calc_i][calc_j]+gamma*max-actionvalue[curr_state-1][action])*eligibility[i][j];

		curr_i = calc_i;					//New State
		curr_j = calc_j;

		time++;
		//epsilon = 1/time;
	/*cout << count <<"Done"<< result << endl;	*/
	}

	//reward[count]+= result;
	error[trial][count] = result;

}
trial++;
}
double sum=0.0,sum_error,mean[episodes],std_dev[episodes];
	for(int i=0;i<episodes;i++){
		sum=0.0,sum_error=0.0;
		for(int j=0;j<trials;j++){
			sum+=error[j][i];
		}
		mean[i]=sum/trials;
		for(int j=0;j<trials;j++)
			sum_error+=pow(error[j][i]-mean[i],2);
		
		sum_error/=(trials-1);
		// sum_error = pow(sum_error,0.5);
		std_dev[i] = sqrt(sum_error);
	}
	for(int i =0;i<episodes;i++){
		cout<< mean[i] << " " << i <<" "<< std_dev[i]<< endl;
	//	sum+=reward[i];
	}

	/*for (int i=0;i<gridworld_size;i++){
		for(int j=0;j<gridworld_size;j++)
			cout << statevalue[i][j] << "\t";
		cout << "\n";
	}*/
	
	return 0;
}
