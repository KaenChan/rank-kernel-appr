#include "fenchel_dual_svm.h"
#include <limits>

#define LOG2(x) 1.4426950408889634*log(double(x)) 
#define MAX_LABEL_NUM 5

//sorting according score
bool compare_score(const struct score_node &m1, const struct score_node &m2) 
{      
     return   m1.score   >   m2.score;
}

//read model
void pairwise_fenchel_dual_svm::read_model()
{
	std::fstream model_stream;
	model_stream.open(modelFileName.c_str(), std::fstream::in);
	if (!model_stream) {
		std::cerr << "Error opening model input file " << modelFileName << std::endl;
		exit(1);
	}

	std::cerr << "Reading model from: " << modelFileName << std::endl;
	string model_string;
	std::getline(model_stream, model_string);
	model_stream.close();

	std::stringstream count_stream(model_string);
	float w;
	int dim =0;
	while (count_stream >> w) { 
		++dim;
	}
    
   // Allocate weights.
   weight = new double[dim];
   if (weight == NULL) {
	 std::cerr << "Not enough memory for weight vector of dimension: " 
		  <<  dim << std::endl;
	 exit(1);
   }
  
  // Fill weights from weights in string.
  std::stringstream weight_stream(model_string);
  for (int i = 0; i < dim; ++i) {
	weight_stream >> weight[i];
  }
}


//read the ranking data
void pairwise_fenchel_dual_svm::read_data()
{
	cout << "Reading file: " << inputFileName  << endl;
	FILE *fpin;
	fpin = fopen(inputFileName.c_str(), "r");
	if (fpin == NULL)
	{
		std::cerr << "Cannot open input file: " << inputFileName.c_str() << "\n";
		system("pause");
		exit(1);
	}  
	m = 0;
	d = 0;
	nonzero = 0;   
	
	while (1)
	{
		int c = fgetc(fpin);
		switch(c)
		{
		case '\n':
			++m; 
			break;
		case ':': 
			++nonzero;
			break;
		case EOF:
			goto out;
		default:
			;
		}
	}
out:  
	rewind(fpin);
	nonzero-=m;
	value = new double[nonzero];
	cIndex = new int[nonzero];
	rIndex = new int[m+1];
	queryId = new int[m];
	label=new int[m];

	int j=0;

	for(int i=0;i<m;i++)
	{
			int t=1;
			rIndex[i]=j;

			while(1)
			{
				int c;
				do {
					c = getc(fpin);
					if(c=='\n') goto out2;
				} while(isspace(c));

				ungetc(c,fpin);
				if (c=='#')
				{
					do
					{
						c= getc(fpin);
					}while(c!='\n');
						goto out2;
				}
				else if(1==t)
				{
					fscanf(fpin,"%d",&label[i]);
					t=2;
				}					
				else if(2==t)
				{

					if(!fscanf(fpin,"qid:WT04-%d ", &queryId[i]))
						fscanf(fpin,"%d ", &queryId[i]);
					t=0;
				}
				else
				{
					fscanf(fpin,"%d:%lf",&cIndex[j],&value[j]);
					cIndex[j]=cIndex[j]-1;                         
					if(cIndex[j]>d) d=cIndex[j];                       
					++j;
					if(j>nonzero) break;
				}
			}	 
	out2:
			;
	}
	rIndex[m]=nonzero;
	d++;
	#if BIAS
		d++;
	#endif

	fclose(fpin);
	
	return;

}



//obtain the index of the queries
void pairwise_fenchel_dual_svm::getQueryInd()
{
	nq=1;
	for (int i=0;i<m;i++)
	{
		if (i<m-1 && queryId[i]!=queryId[i+1])	//same query
		{
			nq++;
		}
	}

	qind=new int[nq+1];
	qind[0]=0;
	nq=1;
	for (int i=0;i<m;i++)
	{
		if (i<m-1 && queryId[i]!=queryId[i+1])	//same query
		{
			qind[nq++]=i+1;
		}
	}
	qind[nq]=m;
}

//write model
void  pairwise_fenchel_dual_svm::write_model()
{
	cout << "Writing model: " << outputFileName  << endl;
	
	FILE *fpin;
	fpin = fopen(outputFileName.c_str(), "w");
	if (fpin == NULL)
	{
		std::cerr << "Cannot open output file: " << outputFileName.c_str() << "\n";
		system("pause");
		exit(1);
	}  

	for(int i=0;i<d;i++)
	{
		fprintf(fpin,"%lf ",weight[i]);
	}
	fclose(fpin);
	cout<<"Writing model done!"<<endl; 
}

//write the socre file.
void  pairwise_fenchel_dual_svm::prediction()
{
	read_model();
	FILE *fpin;
	fpin = fopen(outputFileName.c_str(), "w");
	if (fpin == NULL)
	{
		std::cerr << "Cannot open output file: " << outputFileName.c_str() << "\n";
		system("pause");
		exit(1);
	}  
	
	for(int i = 0; i < m; i++)
	{
		double wxb = 0.0;
		for (int l=rIndex[i];l<rIndex[i+1];l++)
			wxb += weight[cIndex[l]]*value[l];
		
		fprintf(fpin,"%lf \n",wxb);
	}
	fclose(fpin);
	cout << "Writing score : " << outputFileName  << "done!"<<endl;
}




void pairwise_fenchel_dual_svm::getPairInd()
{
	np=0;
	pind=new int[nq+1];
	pind[0]=0;
	for (int i=0;i<nq;i++)
	{
		for (int j=qind[i];j<qind[i+1]-1;j++)
			for (int k=j+1;k<qind[i+1];k++)
			{
				if(label[j]!=label[k])
					np++;
			}
		pind[i+1]=np;
	}	
}
//init
void pairwise_fenchel_dual_svm::init()
{
	weight=new double[d];
	getQueryInd();
	getPairInd();
	score=new double[np];
	dist=new double[np];

	for(int i=0;i<np;i++)
	{
		//dist[i]=2.0/(np*r);
		dist[i] = 2.0*radius/np;
		score[i]=0;
	}
	for(int i=0;i<d;i++)
	{
		weight[i]=0;
	}
}

 //Greedily choose a feature to update
 int pairwise_fenchel_dual_svm::chooseMax(int& maxC,double& maxValue)
 {
 	double* r=new double[d];
 	memset(r,0,sizeof(double)*d);
	int sign = 1;
 
 	int pNum=0;
 	int pLabel=0;
 	for (int i=0;i<nq;i++)
 	{
 		for (int j=qind[i];j<qind[i+1]-1;j++)
 		{
 			for (int k=j+1;k<qind[i+1];k++)
 			{
 				if(label[j]!=label[k])
 				{					
 					pLabel=(label[j]>label[k])?1:-1;
 					for (int l=rIndex[j];l<rIndex[j+1];l++)
 						r[cIndex[l]]+=pLabel*value[l]*dist[pNum];
 					for (int l=rIndex[k];l<rIndex[k+1];l++)
 						  r[cIndex[l]]-=pLabel*value[l]*dist[pNum];

					 #if BIAS
						r[d-1] += pLabel*dist[pNum]; //bias 
					 #endif
 					 pNum++;
					 
 				  }
 				 
 			  }
 		}
 		
 	}
 
 	maxValue=-100000000;
 	for(int i=0;i<d;i++)
 	{
 		if(r[i]>maxValue)
 		{
 			maxC=i;
 			maxValue=r[i];
			sign = 1;
 		}
		else if(-r[i] > maxValue)
		{
			maxC=i;
 			maxValue = -r[i];
			sign = -1;
		}
 	}
   delete[]r;
   return sign;
 }


void pairwise_fenchel_dual_svm::getFeature(int c,double *cFeature)
{

	int pNum=0;
	int pLabel=0;
	for (int i=0;i<nq;i++)
	{
		for (int j=qind[i];j<qind[i+1]-1;j++)
		{
			for (int k=j+1;k<qind[i+1];k++)
			{
				if(label[j]!=label[k])
				{

					pLabel=(label[j]>label[k])?1:-1;
					
					#if BIAS
						//choose the bias
						if(c == d-1)
						{
							cFeature[pNum]=pLabel;
						}

						else
					#endif
						{
							int l=rIndex[j];
							int r=rIndex[j+1]-1;					
							while(l<=r)
							{
								int mid=(l+r)/2;
								if(cIndex[mid]<c)
									l=mid+1;
								else if(cIndex[mid]>c)
									r=mid-1;
								else
								{
									cFeature[pNum]+=pLabel*value[mid];
									break;
								}
							}

							l=rIndex[k];
							r=rIndex[k+1]-1;					
							while(l<=r)
							{
								int mid=(l+r)/2;
								if(cIndex[mid]<c)
									l=mid+1;
								else if(cIndex[mid]>c)
									r=mid-1;
								else
								{
									cFeature[pNum]-=pLabel*value[mid];
									break;
								}
							}
						}
					pNum++;
				  }
				 
			  }
		}
	}

}


//min  sum(|1-Kw|+)^2)  s.t |w|1 < r
void pairwise_fenchel_dual_svm::learn()
{
	int maxC;
	double maxValue;
	double* cFeature=new double[np];

	double sumDiff;
	double maxDiff;
	double eta;

	int iter=1;
	double start = clock();
	while(1)
	{
		//Greedily choose a feature to update 
		int sign = chooseMax(maxC,maxValue);
		memset(cFeature,0,sizeof(double)*np);
		getFeature(maxC,cFeature);

		//check if the early stopping criterion is satisfied
		maxDiff=0;
		sumDiff=0;
		for(int i=0;i<np;i++)
		{
			double diff=(cFeature[i]*sign-score[i]);						
			maxDiff += diff *diff;			
			sumDiff+=dist[i]*diff;			
		}
		
		//cout<<sumDiff<<endl;

		if(sumDiff<=error)
		{
			cout<<"iter:"<<iter<<endl;
			break;
		}
	
		//Compute an appropriate step size
		eta = compute_step(sign,maxC,cFeature,np);

		if(iter > max_iter || eta == 0)
		{
			cout<<"iter:"<<iter<<endl;
			break;
		}
		
		//Update the model with the chosen feature and step size
		for(int i=0;i<d;i++)
		{
			if(i==maxC)
			{
				weight[i]=(1-eta)*weight[i]+eta*sign;
			}
			else
			{
				weight[i]=(1-eta)*weight[i];
			}
		}
		
		//update the primal variable : dist
		for(int i=0;i<np;i++)
		{
			score[i]=(1-eta)*score[i]+eta*cFeature[i]*sign;
			
			if(1.0/radius - score[i] > 0){
				dist[i] = (1.0/radius - score[i])*2.0*radius*radius/np;
			}
			else
				dist[i] = 0;
		}
		
	
		iter++;
	}
	delete[]cFeature;
}

//Compute an appropriate step size
double pairwise_fenchel_dual_svm::compute_step(int sign,int index,double* cFeature,int number)
{
	//min sum (max(0,1-y_i * <(1-eta)w + eta*sign*e^j, x_i>)^2  s.t 0 <= eta <= 1
	//let a_i = y_i * <sign*e^j-w,x_i> b_i = r - y_i *<w,x_i>
	//that is equal to :
		//min sum(max(0,b_i - a_i*eta))^2   s.t 0 <= eta <= 1

	double *a = new double[number]; //a_i = y_i * <sign*e^j-w,x_i> 
	double *b = new double[number]; //b_i = r - y_i *<w,x_i>
	double b_square_neg = 0.0;
	double a_square_neg = 0.0;
	double ab_neg = 0.0;
	double b_square_pos = 0.0;
	double a_square_pos = 0.0;
	double ab_pos = 0.0;
	double *tempEta = new double[number];
	vector<score_node> scoreList;

	for(int i=0; i<number; i++)
	{
		a[i] = cFeature[i]*sign-score[i];
		//b[i] = r - score[i];
		b[i] = 1.0/radius - score[i]; 
		if(a[i] == 0)
			continue;
		//let b_i - a_i*tempEta_i = 0
		tempEta[i] = b[i] /a[i];

		// 0 <= eta <= 1
		if( tempEta[i] < 1 && tempEta[i] >0)
		{
			score_node node;
			node.index = i;
			node.score = tempEta[i];
			scoreList.push_back(node);
		}

		if(tempEta[i] < 1 && a[i] < 0)
		{
			b_square_neg += b[i]*b[i];
			a_square_neg += a[i]*a[i];
			ab_neg += a[i]*b[i];
		}

		if(tempEta[i] >= 1 && a[i] > 0)
		{
			b_square_pos += b[i]*b[i];
			a_square_pos += a[i]*a[i];
			ab_pos += a[i]*b[i];
		}
	}

	//sort
  	sort(scoreList.begin(),scoreList.end(),compare_score);

   double best_eta;
   double min_loss = numeric_limits<double>::max() ;
   if(scoreList.size() == 0)
   {
	   best_eta = (ab_pos + ab_neg) /(a_square_neg + a_square_pos);
	   if(best_eta < 0 || best_eta > 1)
	   {
		   if(fabs(0 - best_eta) < fabs(1 - best_eta))
				best_eta = 0;
		   else
			    best_eta = 1;
	   }
   }
   else
   {
	   double current_eta;
	   double current_loss;
		for(int i=0; i<=scoreList.size(); i++)
		{
			double start=0;
			double end=1;
			if(i == 0)
			{
				start = scoreList[i].score;
				end = 1;
			}
			else if(i == scoreList.size())
			{
				start = 0;
				end = scoreList[i-1].score;
			}
			else
			{
				start = scoreList[i].score;
				end = scoreList[i-1].score;
			}
			if( i-1 >= 0)
			{
				if(a[scoreList[i-1].index] < 0)
				{
					b_square_neg -= b[scoreList[i-1].index]*b[scoreList[i-1].index];
					a_square_neg -= a[scoreList[i-1].index]*a[scoreList[i-1].index];
					ab_neg -= a[scoreList[i-1].index]*b[scoreList[i-1].index];
				}
				else if(a[scoreList[i-1].index] > 0)
				{
					b_square_pos += b[scoreList[i-1].index]*b[scoreList[i-1].index];
					a_square_pos += a[scoreList[i-1].index]*a[scoreList[i-1].index];
					ab_pos += a[scoreList[i-1].index]*b[scoreList[i-1].index];
				}
			}

			if(start >= end)
				continue;

			 current_eta = (ab_pos + ab_neg) /(a_square_neg + a_square_pos);
			 if(current_eta < start || current_eta > end)
			 {
				 if(fabs(start - current_eta) < fabs(end - current_eta))
					current_eta = start;
				 else
					current_eta = end;
			 }
			
			current_loss = b_square_neg - 2*ab_neg*current_eta + a_square_neg*current_eta*current_eta;
			current_loss += b_square_pos - 2*ab_pos*current_eta + a_square_pos*current_eta*current_eta;
			if(current_loss < min_loss)
			{
				best_eta = current_eta;
				min_loss = current_loss;
			}
		}

   }
   delete []a;
   delete []b;
   delete []tempEta;
   return best_eta;
}



pairwise_fenchel_dual_svm::~pairwise_fenchel_dual_svm()
{	

	if(rIndex!=NULL)
		delete[]rIndex;
	if(cIndex!=NULL)
		delete[]cIndex;
	if(value!=NULL)
		delete[]value;	
	if(qind!=NULL)
		delete[]qind;
	if(pind!=NULL)
		delete[]pind;
	if(dist!=NULL)
		delete[]dist;
	if(score!=NULL)
		delete[]score;
	if(weight!=NULL)
		delete[]weight;
	if(queryId!=NULL)
		delete[]queryId;
	if(label!=NULL)
		delete[]label;

}

