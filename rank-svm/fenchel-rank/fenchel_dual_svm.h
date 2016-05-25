#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include <map>
#include <iostream>
#include <limits>
#include <fstream>
#include <sstream> 
#include <ctime>
// #include <mex.h>

using namespace std;

#define BIAS 0


class pairwise_fenchel_dual_svm
{
public:
	pairwise_fenchel_dual_svm(string modelFile,string inFile,string outFile)
	{
		label=NULL;
		queryId=NULL;
		pind=NULL;
		qind=NULL;
		inputFileName=inFile;
		outputFileName=outFile;
		score=NULL;
		weight=NULL;
		dist=NULL;
		modelFileName = modelFile;
		read_data();
		init();
	}
	pairwise_fenchel_dual_svm(string inFile,double e,string outFile,double r,int max_iter)
	{
		label=NULL;
		queryId=NULL;
		pind=NULL;
		qind=NULL;
		inputFileName=inFile;
		error=e;
		outputFileName=outFile;
		score=NULL;
		weight=NULL;
		dist=NULL;
		this->radius = r;
		this->max_iter = max_iter;
		read_data();
		init();
	}
	pairwise_fenchel_dual_svm(double* value,int* label,int* qids,int* ridx,int* cidx,double e,double r,int max_iter,int m, int d)
	{
		this->label=label;
		this->queryId=qids;
		this->cIndex=cidx;
		this->value=value;
		this->rIndex=ridx;
		pind=NULL;
		qind=NULL;
		error=e;
		score=NULL;
		weight=NULL;
		dist=NULL;
		this->radius = r;
		this->max_iter = max_iter;
		this->m = m;
		this->d = d;
		nonzero = m*d;
		init();
	}
public:	
	void learn();
	void write_model();
	void read_model();
	void prediction();
	 ~pairwise_fenchel_dual_svm();

	double* get_w()
	{
		return weight;
	}

private:	
	int chooseMax(int &maxC,double& maxValue);
	void getFeature(int c,double *cFeature);
	void init(); 
	void read_data();
	void getQueryInd();	
	void getPairInd();
	double compute_step(int sign,int index,double* cFeature,int number);


 private:
    string inputFileName;
	string outputFileName;                                            
	double error;

	int* rIndex;
	int* cIndex;
	double* value;
    double nonzero;
	int m;
	int d;
	int *label;//label
	int *queryId;//query id

	double* score;
	double* weight;
		
	int nq;
	int* qind;
	
	double* dist;


	int np;
	int* pind;

	double radius; //||w|| <= radius
	double max_iter;

	string modelFileName;//model

};


struct score_node      
{
	double score;      
	int index;
};   

