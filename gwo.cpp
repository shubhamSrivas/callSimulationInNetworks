// objf is the objective function which needs to be defined
vector<double> GWO(int lb,int ub,int dim,int SearchAgentsNo,int max_iter){
    vector<int> alpha_pos(dim,0);
    double alpha_score=DBL_MAX;
    vector<int> beta_pos(dim,0);
    double beta_score=DBL_MAX;
    vector<int> delta_pos(dim,0);
    double delta_score=DBL_MAX;
    vector<int> lb_vec(dim,lb);
    vector<int> ub_vec(dim,ub);
    vector<vector<int>> positions(SearchAgentsNo,vector<int>(dim));
    for(int i=0;i<dim;i++){
        for(int j=0;j<SearchAgentsNo;j++){
            srand(time(0));
            double rand_no=((double)rand())/RAND_MAX;
            positions[j][i]=rand_no*(ub_vec[i]-lb_vec[i])+lb_vec[i];
        }
    }
    vector<double> Convergence_curve(max_iter);
    for(int l=0;l<max_iter;l++){
        for(int i=0;i<SearchAgentsNo;i++){
            for(int j=0;j<dim;j++){
                if(positions[i][j]<lb_vec[j])
                    positions[i][j]=lb_vec[j];
                else if(positions[i][j]>ub_vec[j])
                    positions[i][j]=ub_vec[j];
            }
            //objf is the objective function which needs to be minimized
            double fitness;
            fitness=objf(positions[i]);
            if(fitness<alpha_score){
                alpha_score=fitness;
                for(int j=0;j<dim;j++)alpha_pos[j]=positions[i][j];
            }
            if(fitness>alpha_score && fitness<beta_score){
                beta_score=fitness;
                for(int j=0;j<dim;j++)beta_pos[j]=positions[i][j];
            }
            if(fitness>alpha_score && fitness>beta_score && fitness<delta_score){
                delta_score=fitness;
                for(int j=0;j<dim;j++)delta_pos[j]=positions[i][j];
            }
        }
        double a=2-(double)l*(2./(double)max_iter);
        for(int i=0;i<SearchAgentsNo;i++){
            for(int j=0;j<dim;j++){
                srand(time(0));
                double r1=((double)rand())/RAND_MAX;
                double r2=((double)rand())/RAND_MAX;
                double a1=2*a*r1-a;
                double c1=2*r2;
                double d_alpha=abs(c1*alpha_pos[j]-positions[i][j]);
                double x1=alpha_pos[j]-a1*d_alpha;
                r1=((double)rand())/RAND_MAX;
                r2=((double)rand())/RAND_MAX;
                double a2=2*a*r1-a;
                double c2=2*r2;
                double d_beta=abs(c2*beta_pos[j]-positions[i][j]);
                double x2=beta_pos[j]-a2*d_beta;
                r1=((double)rand())/RAND_MAX;
                r2=((do
                uble)rand())/RAND_MAX;
                double a3=2*r1*a-a;
                double c3=2*r2;
                double d_delta=abs(c3*delta_pos[j]-positions[i][j]);
                double x3=delta_pos[j]-a3*d_delta;
                positions[i][j]=(x1+x2+x3)/3;
            }
        }
        Convergence_curve[l]=alpha_score;
        cout<<"At iteration "<<l<<" the best fitness is "<<alpha_score;
    }
    return Convergence_curve;
}
// class solution:
//     def __init__(self):
//         self.best = 0
//         self.bestIndividual=[]
//         self.convergence = []
//         self.optimizer=""
//         self.objfname=""
//         self.startTime=0
//         self.endTime=0
//         self.executionTime=0
//         self.lb=0
//         self.ub=0
//         self.dim=0
//         self.popnum=0
//         self.maxiers=0