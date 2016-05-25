
//
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include "fenchel_dual_svm.h"
using namespace std;

// --- options

string trainfile;
string modelfile;
string resultfile;
double err = 0.001;
int T = 1000;
double r = 1.0;
int flag = 1;

void usage()
{
  cerr << "Usage of train: fenchel_dual_svm -flag 1 [options] trainfile modelfile" << endl
       << "Options:" << endl
       << " -r <r : the constrain of ||w|| <= r>" << endl
       << " -T <T: the maximum number of iterations>" << endl
       << " -err <err : the required accuray>" << endl
       << " -flag <flag : " << endl
       << "            flag = 1 : ranking" << endl
       << "            flag = 2 : test >" <<endl
       << endl;
  cerr << "Usage of test: fenchel_dual_svm -flag 2 testfile modelfile scorefile" << endl
       << " -flag <flag : " << endl
       << "            flag = 1 : ranking" << endl
       << "            flag = 2 : test >" <<endl
       << endl;
  exit(10);
}


void parse(int argc, char* argv[])
{
    
  for (int i=1; i<argc; i++)
    {
      const char *arg = argv[i];
      if (arg[0] != '-')
        {
          if (trainfile.empty())
            trainfile = arg;
          else if (modelfile.empty())
            modelfile = arg;
          else if (resultfile.empty())
            resultfile = arg;
          else
            usage();
        }
      else
        {
          while (arg[0] == '-') arg += 1;
          string opt = arg;
          if (opt == "r" && i+1<argc)
            {
              r = atof(argv[++i]);
              cout << "Using r=" << r << "." << endl;
              assert(r>0);
            }
          else if (opt == "T" && i+1<argc)
            {
              T = atoi(argv[++i]);
              cout << "Using T = " << T << " iterations." << endl;
              assert(T>0);
            }
          else if (opt == "flag" && i+1<argc)
            {
              flag = atoi(argv[++i]);
              assert(flag == 1 || flag ==2);
            }
          else if (opt == "err" && i+1<argc)
            {
              err = atof(argv[++i]);
              cout << "Using err=" << err << "." << endl;
              assert(err >=0 );
            }
          else
            usage();
        }
    }
  if (trainfile.empty() || modelfile.empty() )
      usage();
  if(flag == 2 && resultfile.empty())
      usage();
 
}


void main(int argc,  char* argv[])
{

    parse(argc,argv);
    switch(flag)
    {
        //ranking
    case 1:
        {           
            pairwise_fenchel_dual_svm sparse_svm(trainfile,err,modelfile,r,T);  
            cout<<"training...."<<endl;
            sparse_svm.learn();
            sparse_svm.write_model();   
        }   
        break;
    case 2:  //prediction
        {
            pairwise_fenchel_dual_svm test(modelfile,trainfile,resultfile);
            test.prediction();  
        }
        break;
    default:
        ;
        
    }
    return;
}


