#include <iostream>
#include<vector>
#include <iomanip>
#include <cmath>
#define ll long double

using namespace std;

ll maxCost=999;
ll minCost=99;
ll resetCost=990;
int TotalStates=15;
int iterations=500;
ll discountFactor=0.7;
vector<ll>stateCost;
vector<ll> expected_optimal_vector;
vector<int> optimalPolicy;


long double probability_cal(long int n, long int i, long int j) {
    long double num = (long double)n + i - j;
    long double den = (long double)n * n +
                      (long double)n * i -
                      ((long double)n * (n + 1)) / 2.0L;
    return num / den;
}

void fill_stateCost(){
    ll slope = (double)(maxCost-minCost)/(double)(TotalStates-1);
    ll intercept = minCost-slope;
    for(int i=1;i<=TotalStates;i++){
        stateCost[i]= slope*i + intercept;
    }
}

ll discountedFutureCost(long int currentState){
    ll result=0;
    for(int futureState=1;futureState<=TotalStates;futureState++){
        result= result+ probability_cal(TotalStates,currentState, futureState)*expected_optimal_vector[futureState];
    }
    return result * discountFactor;
}

void fill_expectedOptimalVector(){
    for(int iter=0;iter<iterations;iter++){
        vector<ll>temp(TotalStates+1);
        for(int i=1;i<=TotalStates;i++){
            ll cost1 = resetCost+stateCost[1]+discountedFutureCost(1);
            ll cost2 = stateCost[i]+discountedFutureCost(i);
            temp[i]=min(cost1,cost2);
        }
        expected_optimal_vector= temp;
    }
}

void fill_optimalPolicy(){
    for(int i=1;i<=TotalStates;i++){
        ll cost1 = resetCost+stateCost[0]+discountedFutureCost(0);
        ll cost2 = stateCost[i]+discountedFutureCost(i);
        //if(abs(cost1-expected_optimal_vector[i])<=abs(cost2-expected_optimal_vector[i])){
         //   optimalPolicy[i]=0;
        //}
        //else optimalPolicy[i]=1;
        
        if(cost1<=cost2)optimalPolicy[i]=0;
        else optimalPolicy[i]=1;
    }
}



void policy_iteration(){
    for(int iter=0;iter<iterations;iter++){
        vector<ll>temp(TotalStates+1);
        for(int i=1;i<=TotalStates;i++){
            if(optimalPolicy[i]==0){
                temp[i]=resetCost+stateCost[0]+discountedFutureCost(0);
            }
            else{
                temp[i]=stateCost[i]+discountedFutureCost(i);
            }
        }
        expected_optimal_vector = temp;
    }
    fill_optimalPolicy();
}


int main() {

    stateCost.resize(TotalStates+1);
    expected_optimal_vector.resize(TotalStates+1,0);
    optimalPolicy.resize(TotalStates+1);
    fill_stateCost();
    //fill_expectedOptimalVector();
    //fill_optimalPolicy();
    for(int iter=0;iter<iterations;iter++){
        policy_iteration();
    }
    cout<<"optimal policy "<<endl;
    for(int i=1;i<=TotalStates;i++){
        cout<<optimalPolicy[i]<<" ";
    }

    cout<<"optimal values "<<endl;
    for(int i=1;i<=TotalStates;i++){
        cout<<expected_optimal_vector[i]<<" ";
    }
    cout<<endl;
    cout << "done\n";
    return 0;
}