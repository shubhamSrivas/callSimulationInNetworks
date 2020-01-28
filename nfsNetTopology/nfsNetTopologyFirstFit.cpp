#include <iostream>
#include <stdlib.h>
#include <vector>
#include <queue>
#include <iomanip>
#include <limits>
#include <time.h>
#include <math.h>
#include <cmath>
#include <list>
#include <climits>

using namespace std;

//define a weighted edge as pair of destination and weight(directed)
typedef pair<int,float> Edge ;

//define adjacency list as a vector of vector of edges so as to access particular index as well as add/remove nodes at runtime
typedef vector< vector< Edge > >  AdjacencyList;

typedef vector <vector <int> > wl_slot; //define a wavelength and slot available pair

class Comparator
{
 public:
 int operator() ( const Edge &p1, const Edge &p2)
 {
 return p1.second>p2.second;
 }
};

class Graph
{
    public:
         AdjacencyList MST; //Adjacency list to store MST
         int nodes;    // No. of nodes

         AdjacencyList adjList;
         Graph(int vertices);// Constructor
         void addEdge(int source, int destination ,float weight);   // to add an edge to graph
         void printGraph();//to display the generated graph

         //Create random graph given graph object
         void construct_graph();
         //MST returned as vector of vector of Edges using Prim's algorithm
         void Prim();
};

// WDM class for supporting required functions
class WDM : public Graph
{
    public:
        AdjacencyList Network;//Adjacency list to store topology generated
        int num_wavelengths;//number of wavelengths in each channel
        int num_slots;//number of slots per wavelength
        float slot_width;
        vector< AdjacencyList > wavelength_matrix;//matrix of wavelengths occupied for each channel in topology
        vector< vector <AdjacencyList> > slot_arrays;//slot arrays for each channel (wavelength) and link

        //------------------------------------------------------------------------------------------------------//
        vector< AdjacencyList > wavelength_matrix_new; //matrix of wavelengths for new algo
        vector< vector <AdjacencyList> > slot_arrays_new; //slot arrays for new algo
        wl_slot slot_index; //slot index matrix
        //-----------------------------------------------------------------------------------------------------//
        //vector< vector <int> > traffic_matrix;//traffic matrix showing call requests from source to destination; USING VECTORS AS UNABLE TO USE THIS->NODES
        vector<double> call_end_time;//vectors to store end time for call requests
        vector<double> call_end_time_new;

        WDM(int nodes): Graph(nodes) {}//constructor

        
        //shortest path routing b/w source->destination (stored as vector of nodes) displayed using Dijkstra's algorithm
         vector<Edge> Dijkstra(int source,int destination);

        //Network topology generated using density factor
        void NWGen();

        ///Check if path exists between given source 7 destination nodes
        bool path_exists(int source, int destination);
        //Set number of wavelengths
        void set_wavelength_number(int number);
        //Set number of slots per wavelengths
        void set_slot_number(int number);
        //Set slot's width per slot
        void set_slot_width(float number);
        //Initialise empty wavelegth matrix
        void initialize_wavelength_matrix();
        void initialize_wavelength_matrix_new();
        //print wavelength matrix
        void print_wavelength_matrix();
        //Create empty slots in each link and channel
        void create_empty_slots();
        //Print the slot arrays
        void print_slot_arrays();
        //Initialize traffic matrix (empty)
        void initialize_traffic_matrix();
        //Print traffic matrix
        void print_traffic_matrix();

        //simulate call arrival process
        vector<float> call_process();
        //generate time after which next call occurs
        float next_call_time(double lambda);
        //get wavelength according to first-fit algorithm
        int first_fit(int source,int destination,int slots_required,int request_number);

        //----------------------------------------------------------------------------------------//
        //to push wavlength and available slot in slot_index matrix
        void push(int w,int s,wl_slot &slot_index);

        void merge_inc(wl_slot &slot_index,int l,int m,int r);
        void merge_dec(wl_slot &slot_index,int l,int m,int r);

        //sort slot_index
        void sort(wl_slot &slot_index,int l,int r,int order);

        //get wavelength according to first-last-exact-fit algorithm
        int first_last_fit(int source,int destination,int slots_required,int request_number);

        //function to deallocate slots for call which has been serviced
        void free_serviced_call_slots(double time,int calls_till_now);

        //TODO
        //Yen's k-shortest path algorithm b/w source & destination (see wikipedia for details)
        AdjacencyList k_shortest_path_routing(int source,int destination,int k);
};

//define constructor for Graph {to allocate memory for the graph}
Graph::Graph(int vertices)
{
    adjList.resize(vertices);
    this->nodes=vertices;
}

//add Edge to directed graph
void Graph::addEdge(int source, int destination ,float weight)
{ adjList[source].push_back(make_pair(destination,weight));
  adjList[destination].push_back(make_pair(source,weight)); //duplex network
}

//Add random edges to graph object
void Graph::construct_graph()
{
  int i,source,destination,weight,edges;

  edges = 21;//No. of edges. 

  cout<<edges;
    this->addEdge(0,1,1100);
    this->addEdge(0,2,1600);
    this->addEdge(0,7,2800);
    this->addEdge(1,2,600);
    this->addEdge(1,3,1000);
    this->addEdge(2,5,2000);
    this->addEdge(3,4,600);
    this->addEdge(3,10,2400);
    this->addEdge(4,5,1100);
    this->addEdge(4,6,800);
    this->addEdge(5,9,1200);
    this->addEdge(5,12,2000);
    this->addEdge(6,7,700);
    this->addEdge(7,8,700);
    this->addEdge(8,9,900);
    this->addEdge(8,11,500);
    this->addEdge(8,13,500);
    this->addEdge(10,11,800);
    this->addEdge(10,13,800);
    this->addEdge(11,12,300);
    this->addEdge(12,13,300);
}

void Graph::printGraph()
{
    for(int i=0;i<this->nodes;i++)
    {
        {
        cout<<i<<": ";
        for(int j=0;j<adjList[i].size();j++)
            cout<<"->"<<adjList[i][j].first<<"("<<adjList[i][j].second<<")"<<" ";
        }
        cout<<endl;
    }
}

//Prim's Minimum Spanning Tree implementation
void Graph::Prim() {

    MST.resize(this->nodes);
    priority_queue<pair <int,int>, vector <pair<int,int> >, greater<pair <int,int> > > pq;

    int src;
    for(int i=0;i<this->nodes;++i)
        if(adjList[i].size()){
            src=i;// Taking vertex i as source
            break;
        }

    //vector to store shortest distance from nodes already in MST
    vector<int> key(this->nodes, INT_MAX);

    //vector to store parent along shortest path
    vector<int> parent(this->nodes, 0);

    vector<bool> inMST(this->nodes, false);

    pq.push(make_pair(0, src));
    key[src] = 0;//mark source distance as 0
    while (!pq.empty()) {

        int u = pq.top().second;
        pq.pop();//remove the just extracted node from MinHeap

        inMST[u] = true;  // Include vertex in MST

        vector< pair<int, float> >::iterator i;
        for (i = adjList[u].begin(); i != adjList[u].end(); ++i)//loop to update shortest distance from already established MST
        {

            int v = (*i).first;
            int weight = (*i).second;

            //  If v is not in MST and weight of (u,v) is smaller
            // than current key of v
            if (inMST[v] == false && key[v] > weight)
            {
                // Updating key of v
                key[v] = weight;
                pq.push(make_pair(key[v], v));
                parent[v] = u;
            }
        }
    }
    //filling up MST adjacency list
    for(int i=1;i<parent.size();i++)
        MST[parent[i]].push_back(make_pair(i,key[i]));


    for(int i=1;i<parent.size();i++)
        cout<<parent[i]<<"->"<<i<<"("<<key[i]<<")"<<endl;

}

//Network Generator for given density factor
void WDM::NWGen()
{
     Network=adjList;
}

bool WDM::path_exists(int source, int destination)
{
  int s,d,j,V,i,node;
  s = source;
  d = destination;
  V = this->nodes;
  AdjacencyList adj = this->Network;

  // Base case
  if (s == d)
    return true;
 
    // Mark all the vertices as not visited
    bool *visited = new bool[V];
    for (j = 0; j < V; j++)
        visited[j] = false;
 
    // Create a queue for BFS
    list<int> queue;
 
    // Mark the current node as visited and enqueue it
    visited[s] = true;
    queue.push_back(s);
 
    while (!queue.empty())
    {
        // Dequeue a vertex from queue and print it
        s = queue.front();
        queue.pop_front();
 
        // Get all adjacent vertices of the dequeued vertex s
        // If a adjacent has not been visited, then mark it visited
        // and enqueue it
        for (i=0;i < adjList[s].size();i++)
        {
            // If this adjacent node is the destination node, then 
            // return true
            node = adjList[s][i].first;
            if (node == d)
                return true;
 
            // Else, continue to do BFS
            if (!visited[node])
            {
                visited[node] = true;
                queue.push_back(node);
            }
        }
    }
     
    // If BFS is complete without visiting d
    return false;
}

//Dijkstra's shortest path implementation
vector<Edge> WDM::Dijkstra(int source,int destination)
{
    vector<Edge> path; //vector to store shortest path
    path.clear(); //clear the path vector of Edges
    vector<float> distance(this->nodes);//vector to store distance from source
    vector<int> parent(this->nodes);//vector to store parent along shortest path
    int u,w;//temporary variables to store node on top of MinHeap
    AdjacencyList adjList = this->Network;

    for(unsigned int i = 0 ;i < this->nodes; i++)
    {
      distance[i] = numeric_limits<float>::max();//initialize distance as infinity
      parent[i] = -1;//initialize parent as -1
    }
    distance[source]=0.0f ;//mark source distance as 0

    priority_queue<Edge, vector<Edge> ,Comparator> MinHeap;
    MinHeap.push(make_pair(source,distance[source])) ;

    while(!MinHeap.empty())
    {
        u=MinHeap.top().first;
        if(u==destination)//destination reached
            break;
        MinHeap.pop();//remove the just extracted node from MinHeap
        for(unsigned i=0;i< adjList[u].size();i++)//for all nodes reachable from popped node
        {
            int v= adjList[u][i].first;
            float w = adjList[u][i].second;
            //update min distance from source for each node and its parent in shortest path
            if(distance[v] > distance[u]+w)
              {
                 distance[v] = distance[u]+w;
                 parent[v] = u;
                 MinHeap.push(make_pair(v,distance[v]));
              }
        }
    }

    
    pair<int,float> p = make_pair(destination,distance[destination]-distance[parent[destination]]);
    path.push_back(p); //push the destination vertex & its parent node,edge into path vector

    //loop to push all nodes,paths in path vector in reverse order
    while(p.first!=source)
    {
      if(parent[p.first]!=source)
         p = make_pair(parent[p.first],distance[p.first]-distance[parent[p.first]]);
      else
         p = make_pair(parent[p.first],distance[p.first]);
      path.push_back(p);
    }

    return path;
}

void WDM::set_wavelength_number(int number)
{this->num_wavelengths = number;}

void WDM::set_slot_number(int number)
{this->num_slots = number;}

void WDM::set_slot_width(float number)
{this->slot_width = number;}

//Create empty wavelength matrix
void WDM::initialize_wavelength_matrix()
{   
    int q,i,j;

    wavelength_matrix.clear();
    wavelength_matrix.resize(this->num_wavelengths);//set size of wavelegth matrix to number of wavelegths

    for(q=0;q<wavelength_matrix.size();q++)//loop till number of wavelegths
    {
        wavelength_matrix[q].resize(this->nodes);//set size of wavelegth q matrix to number of nodes//graph kind of

        for(i=0;i<this->nodes;i++)//inner loop till number of nodes
        {
            wavelength_matrix[q][i].resize(Network[i].size());//set size of wavelegth q matrix for node i to number of links
            //no of destinations
            {
            for(j=0;j<Network[i].size();j++)//second inner loop till number of edges from that node//for no of destinations
                {    
                    wavelength_matrix[q][i][j].first = Network[i][j].first;//initialize destination node from topology
                    wavelength_matrix[q][i][j].second = 0;//intialize qth wavelength as 0(free)
                    //here i is the source and wevlength_matrix.first is destination .second wavelength assigned.
                }
            }
        }
    }

    //----------------------------------------------------------------------------------------------------------------//
    //new_algo matrix initialization

    wavelength_matrix_new.clear();
    wavelength_matrix_new.resize(this->num_wavelengths);//set size of wavelegth matrix to number of wavelegths

    for(q=0;q<wavelength_matrix_new.size();q++)//loop till number of wavelegths
    {
        wavelength_matrix_new[q].resize(this->nodes);//set size of wavelegth q matrix to number of nodes//graph kind of

        for(i=0;i<this->nodes;i++)//inner loop till number of nodes
        {
            wavelength_matrix_new[q][i].resize(Network[i].size());//set size of wavelegth q matrix for node i to number of links
            //no of destinations
            {
            for(j=0;j<Network[i].size();j++)//second inner loop till number of edges from that node//for no of destinations
                {    
                    wavelength_matrix_new[q][i][j].first = Network[i][j].first;//initialize destination node from topology
                    wavelength_matrix_new[q][i][j].second = 0;//intialize qth wavelength as 0(free)
                    //here i is the source and wevlength_matrix.first is destination .second wavelength assigned.
                }
            }
        }
    }
    //---------------------------------------------------------------------------------------------//
}

//Create empty slot arrays
void WDM::create_empty_slots()
{
    int r,q,i,j;

    slot_arrays.clear();
    slot_arrays.resize(this->num_slots);//set size of slot_arrays

    for(r=0;r<slot_arrays.size();r++)
    {
        slot_arrays[r].resize(num_wavelengths);//resize all slot arrays to number of wavelengths

        for(q=0;q<num_wavelengths;q++)//loop till number of wavelegths
        {
            slot_arrays[r][q].resize(this->nodes);//set size of wavelegth q matrix to number of nodes

            for(i=0;i<this->nodes;i++)//inner loop till number of nodes
            {
                slot_arrays[r][q][i].resize(Network[i].size());//set size of wavelegth q matrix for node i to number of links

                {
                for(j=0;j<Network[i].size();j++)//second inner loop till number of edges from that node
                    {
                        slot_arrays[r][q][i][j].first = Network[i][j].first;//initialize destination node from topology
                        slot_arrays[r][q][i][j].second = 0;//intialize qth wavelength as 0(free) 
                    }
                }
            }
        }
    }

    //---------------------------------------------------------------------------------------------------//
    //firts-last-exact-fit

    slot_arrays_new.clear();
    slot_arrays_new.resize(this->num_slots);//set size of slot_arrays_new

    for(r=0;r<slot_arrays_new.size();r++)
    {
        slot_arrays_new[r].resize(num_wavelengths);//resize all slot arrays to number of wavelengths

        for(q=0;q<num_wavelengths;q++)//loop till number of wavelegths
        {
            slot_arrays_new[r][q].resize(this->nodes);//set size of wavelegth q matrix to number of nodes

            for(i=0;i<this->nodes;i++)//inner loop till number of nodes
            {
                slot_arrays_new[r][q][i].resize(Network[i].size());//set size of wavelegth q matrix for node i to number of links

                {
                for(j=0;j<Network[i].size();j++)//second inner loop till number of edges from that node
                    {
                        slot_arrays_new[r][q][i][j].first = Network[i][j].first;//initialize destination node from topology
                        slot_arrays_new[r][q][i][j].second = 0;//intialize qth wavelength as 0(free) 
                    }
                }
            }
        }
    }
}

//Generate call process with source,destination and slot requirements
vector<float> WDM::call_process()
{
  int source,destination,slots_required;
  vector<float> result;
  result.resize(5);

  source = rand()%(this->nodes);
  do {
    destination = rand()%(this->nodes);
  } while (destination == source);

  double temp = this->num_slots * this->slot_width;

  float bandwidth_required = ((double)rand()/((double)RAND_MAX/temp)); // randomly generated required bandwidth_required

  if(fmod(bandwidth_required , this->slot_width) > 0)
	slots_required = (int)(bandwidth_required/this->slot_width) + 1;
  else
  	slots_required = (int)(bandwidth_required/this->slot_width);
  slots_required++;//1 is added for band
  result[1]=source;
  result[2]=destination;
  result[3]=slots_required;
  result[4]=(float)bandwidth_required/(float)(slots_required*this->slot_width); // spectrum efficiency

  cout.precision(5);
  cout<<endl<<"Call request between Source node "<<result[1]<<" & Destination node "<<result[2]<<" for bandwidth "<<bandwidth_required\
  <<" GHz, "<<"no. of slots required " <<result[3]<<endl;
  
  return result;
}

//generate time after which next call occurs
float WDM::next_call_time(double service_rate)
{return -logf(1.0f - (float) random() / (RAND_MAX + 1ll)) / service_rate;}//ll to prevent overflow

//get wavelength according to first-fit algorithm
int WDM::first_fit(int source,int destination,int slots_required,int request_number)
{
  int i,j,origin,end,q,slot_number;
  int p,s,temp_slot_number;
  int cfs;//variable to represent continuous number of free slots
  int last,start;//variable to represent starting & last free slot number
  bool wavelength_available;//boolean to represent wavelength availability on this path
  bool slot_available[num_slots];//boolean array to represent slot availability across path & wavelength

  vector<Edge> path = this->Dijkstra(source,destination);


  //cout<<"Shortest path between source node "<<source<<" & destination node "<<destination<<" is"<<endl;
  cout<<"PATH: ";
  for(i=path.size()-1;i>0;i--)//print the shortest path
      cout<<path[i].first<<"->"<<path[i-1].first<<"("<<path[i].second<<")"<<" ";

  cout<<endl;

  for(q=0;q<this->num_wavelengths;q++)//loop till number of wavelengths
  {
    start=0;
    cfs=0;

    wavelength_available = true;

    for(i=0;i<num_slots;i++)
      slot_available[i] = true;

    for(i=path.size()-1;i>0;i--)//inner loop till number of nodes on path
    {    
        origin = path[i].first;
        end = path[i-1].first;

        for(slot_number=0;slot_number<num_slots;slot_number++)//second inner loop till number of slots in wavelength
            {
                if(slot_arrays[slot_number][q][origin][j].second==0)
                  slot_available[slot_number]=true;
                else
                  slot_available[slot_number]=false;  
            }
    }
    
    for(i=0;i<num_slots;i++)
    {
      if(slot_available[i])
        cfs++;
      else
      {
        start=i+1;
        cfs=0;
      }

      if(cfs==slots_required)
          break;
    }

    if(cfs<slots_required)
      {
        wavelength_available=false;
        cout<<"FF::Wavelength "<<q<<" not available"<<endl<<endl;
        continue;
      }
    
    if(wavelength_available)
    {
      last = start + slots_required -1 ;
      cout<<"FF::Wavelength "<<q<<" is available."<<endl<<endl;
      for(i=path.size()-1;i>0;i--)//inner loop till number of nodes on path
        {    
          origin = path[i].first;
          end = path[i-1].first;

          for(slot_number=start;slot_number<=last;slot_number++)//second inner loop till number of slots in wavelength
              {   
                  for(j=0;j<Network[origin].size();j++)//USELESS LOOP; Only to find the number of edge which contains destination
                      if (slot_arrays[slot_number][q][origin][j].first == end)
                        break;

                  this->slot_arrays[slot_number][q][origin][j].second = request_number;//Mark the slots with the call request it is alloted to                   
              }  
        }
    
      //this->print_slot_arrays();

      return 1;
    }
  }
  
  cout<<endl<<"No possible wavelegth and slot assignment along the shortest path for given source,destination and slot requests"<<endl<<endl;
  return 0;
}

void WDM::push(int w,int s,wl_slot &slot_index){
    vector<int> temp;
    temp.push_back(w);
    temp.push_back(s);

    slot_index.push_back(temp);
}
void WDM::merge_inc(wl_slot &slot_index,int l,int m,int r) {
    int i,j,k;
    wl_slot left;
    left.clear();
    wl_slot right;
    right.clear();

    j = k = 0;
    
    for(i = 0;i <= (m-l);i++)
        left.push_back(slot_index[i+l]);

    for(i = 0;i < (r-m);i++)
        right.push_back(slot_index[i+m+1]);

    for(i = l; i <= r;i++){
        if(j < m-l+1 && k < r-m){
            if((left[j][1] > right[k][1])){
                slot_index[i] = right[k];
                k++;
            }
            else if((left[j][1] <= right[k][1])){
                slot_index[i] = left[j];
                j++;
            }
        }
        else if(j < m-l+1){
            slot_index[i] = left[j];
            j++;
        }
        else {
            slot_index[i] = right[k];
            k++;
        }

    }

}

void WDM::merge_dec(wl_slot &slot_index,int l,int m,int r) {
    int i,j,k;
    wl_slot left;
    left.clear();
    wl_slot right;
    right.clear();

    j = k = 0;
    
    for(i = 0;i <= (m-l);i++)
        left.push_back(slot_index[i+l]);

    for(i = 0;i < (r-m);i++)
        right.push_back(slot_index[i+m+1]);

    for(i = l; i <= r;i++){
        if(j < m-l+1 && k < r-m){
            if((left[j][1] < right[k][1])){
                slot_index[i] = right[k];
                k++;
            }
            else if((left[j][1] >= right[k][1])){
                slot_index[i] = left[j];
                j++;
            }
        }
        else if(j < m-l+1){
            slot_index[i] = left[j];
            j++;
        }
        else {
            slot_index[i] = right[k];
            k++;
        }

    }

}

void WDM::sort(wl_slot &slot_index,int l,int r,int order) {
    if(l < r){
        int m = l+(r-l)/2;

        sort(slot_index,l,m,order);
        sort(slot_index,m+1,r,order);

        if(order == 1)
            merge_inc(slot_index,l,m,r);
        else
            merge_dec(slot_index,l,m,r);
    }
}

int WDM::first_last_fit(int source,int destination,int slots_required,int request_number)
{
  int i,j,origin,end,q,slot_number;
  int p,s,temp_slot_number;
  int cfs;//variable to represent continuous number of free slots
  int last,start;//variable to represent starting & last free slot number
  bool wavelength_available;//boolean to represent wavelength availability on this path
  bool slot_available[num_slots];//boolean array to represent slot availability across path & wavelength

  vector<Edge> path =this->Dijkstra(source,destination);

  //---------------------------------------------------------------------------//
  int cfs_available;
  wl_slot slot_index;


  for(q=0;q<this->num_wavelengths;q++)//loop till number of wavelengths
  {
    cfs_available=0;

    origin = path[path.size()-1].first;
    end = path[path.size()-2].first;

    for(slot_number=0;slot_number<num_slots;slot_number++)//second inner loop till number of slots in wavelength
    {
      if(slot_arrays_new[slot_number][q][origin][j].second==0)
        cfs_available++;
      else
        break;             
    }
    //cout<<"["<<q<<"  "<<cfs_available<<"]"<<"\t";
    push(q,cfs_available,slot_index);
  }

    sort(slot_index,0,slot_index.size()-1,request_number%2);

  //-----------------------------------------------------------------------------------//

  for(int k=0;k<this->num_wavelengths;k++) //to start from minimum slot index
  {
    if(slot_index[k][1] < slots_required)
        cout<<"FLF::Wavelength "<<slot_index[k][0]<<" not available"<<endl;
    else {

    q = slot_index[k][0];
    start=0;
    cfs=0;

    wavelength_available = true;

    for(i=0;i<num_slots;i++)
      slot_available[i] = true;

    for(i=path.size()-1;i>0;i--)//inner loop till number of nodes on path
    {    
        origin = path[i].first;
        end = path[i-1].first;

        for(slot_number=0;slot_number<num_slots;slot_number++)//second inner loop till number of slots in wavelength
            {
                if(slot_arrays_new[slot_number][q][origin][j].second==0)
                  slot_available[slot_number]=true;
                else
                  slot_available[slot_number]=false;  
            }  
    }
    
    for(i=0;i<num_slots;i++)
    {
      if(slot_available[i])
        cfs++;
      else
      {
        start=i+1;
        cfs=0;
      }

      if(cfs==slots_required)
          break;
    }

    if(cfs<slots_required)
      {
        wavelength_available=false;
        cout<<"FLF::Wavelength "<<q<<" not available"<<endl;
        continue;
      }
    
    if(wavelength_available)
    {
      last = start + slots_required -1 ;
      cout<<"FLF::Wavelength "<<q<<" is available."<<endl;
      for(i=path.size()-1;i>0;i--)//inner loop till number of nodes on path
        {    
          origin = path[i].first;
          end = path[i-1].first;

          for(slot_number=start;slot_number<=last;slot_number++)//second inner loop till number of slots in wavelength
              {   
                  for(j=0;j<Network[origin].size();j++)//USELESS LOOP; Only to find the number of edge which contains destination
                      if (slot_arrays_new[slot_number][q][origin][j].first == end)
                        break;

                  this->slot_arrays_new[slot_number][q][origin][j].second = request_number;//Mark the slots with the call request it is alloted to                     
              }  
        }
      return 1;
    }
    }
  }
  cout<<endl<<"No possible wavelegth and slot assignment along the shortest path for given source,destination and slot requests"<<endl;
  return 0;
}

void WDM::free_serviced_call_slots(double time,int calls_till_now)
{
  int call_number,i,q,r,j,calls_completed=0,calls_in_progress=0;

  for(call_number=1;call_number<call_end_time.size();call_number++)
  {
    if(call_end_time[call_number]==-1)//call has been serviced
      calls_completed++;
    else if(call_end_time[call_number]>time)//If the call end time for ith request is after current time then continue
      calls_in_progress++;
    else
      {
        for(r=0;r<slot_arrays.size();r++)//loop till number of slots
          for(q=0;q<num_wavelengths;q++)//loop till number of wavelegths
              for(i=0;i<this->nodes;i++)//inner loop till number of nodes
                    for(j=0;j<Network[i].size();j++)//second inner loop till number of edges from that node
                          if(slot_arrays[r][q][i][j].second == call_number)//if slot has been alloted to that call number
                              slot_arrays[r][q][i][j].second = 0;//free that slot
        call_end_time[call_number] = -1;//mark call as having been serviced
        cout<<"FF::Call "<<call_number<<", completed."<<endl;
      }
  }

  cout<<"FF::Calls completed : "<<calls_completed<<", Calls in progress : "<<calls_in_progress<<endl;

  //----------------------------------------------------------------------------------------------------------------------//

    
  calls_completed=0,calls_in_progress=0;

  for(call_number=1;call_number<call_end_time_new.size();call_number++)
  {
    if(call_end_time_new[call_number]==-1)//call has been serviced
      calls_completed++;
    else if(call_end_time_new[call_number]>time)//If the call end time for ith request is after current time then continue
      calls_in_progress++;
    else
      {
        for(r=0;r<slot_arrays_new.size();r++)//loop till number of slots
          for(q=0;q<num_wavelengths;q++)//loop till number of wavelegths
              for(i=0;i<this->nodes;i++)//inner loop till number of nodes
                    for(j=0;j<Network[i].size();j++)//second inner loop till number of edges from that node
                          if(slot_arrays_new[r][q][i][j].second == call_number)//if slot has been alloted to that call number
                              slot_arrays_new[r][q][i][j].second = 0;//free that slot
        call_end_time_new[call_number] = -1;//mark call as having been serviced
        cout<<"FLF::Call "<<call_number<<", completed."<<endl;
      }
  }

  cout<<"FLF::Calls completed : "<<calls_completed<<", Calls in progress : "<<calls_in_progress<<endl;


  //------------------------------------------------------------------------------------------------------------------//  
}

double vectorSum(vector<float> &v){
  double v_sum = 0;
  for(int i=0;i<v.size();i++){
    // cout<<v[i]<<"|";
    v_sum += v[i];
  }
  return v_sum;
}

int main() {
    int nodes=14,edge_density,density,waves,slots;
    double lambda,mu,Time,width;
    WDM topo(nodes);
    topo.construct_graph();
    cout<<endl<<"The Graph is:"<<endl;
    topo.printGraph();

    cout<<endl<<endl<<"The MST is:"<<endl;
    topo.Prim();
    topo.NWGen();

    //-----------------------------------------------------------------------------------//

    // cout<<"Enter number of wavelengths per link:"<<endl;
    // cin>>waves; 
    topo.set_wavelength_number(1);
    topo.initialize_wavelength_matrix();

    cout<<"Enter number of slots per link:"<<endl;
    cin>>slots; 
    topo.set_slot_number(slots);
    topo.create_empty_slots();

    cout<<"Enter width of slot per link:"<<endl;
    cin>>width; 
    topo.set_slot_width(width);

    //----------------------------------------------------------------------------------------//
   
    cout<<endl<<"Enter call arrival rate (inverse of average duration between calls) lambda:"<<endl;
    cin>>lambda;
    cout<<"Enter service rate mu (inverse of average holding time):"<<endl<<"NOTE: SHOULD BE LESS THAN LAMBDA:"<<endl;
    cin>>mu;
    cout<<"Enter time till when to simulate call arrivals and serivce:"<<endl;
    cin>>Time;

    int calls_blocked,i,call_completed,total_calls = ceil(lambda*Time);
    //------------------------------------------------------------------------------------//
    int call_completed_new, calls_blocked_new;
    //-----------------------------------------------------------------------------------//
    double t_current,t_hold;
    topo.call_end_time.clear(); //vector to store end time for call requests
    topo.call_end_time_new.clear();
    //topo.call_end_time.resize(total_calls);
    

    srand(time(NULL));
    calls_blocked = calls_blocked_new = 0;

    topo.call_end_time.push_back(-1);
    topo.call_end_time_new.push_back(-1);

    //---------------------------------------------------------------------------------------//
    vector<float> temp_call;
    temp_call.resize(4);

    float delay_sum = 0;
    float avg_efficiency = 0;
    float stopping_point = 0;
    int no_of_calls;
    bool stopping_flag = false;
    vector<float> ff_cont_slots;
    vector<float> flf_cont_slots;
    //---------------------------------------------------------------------------------------//

    for(i=1;;i++)
    {
      
      t_current = t_current + topo.next_call_time(lambda);

      if(stopping_flag){ // current time is greater than last serviced call time
        cout<<"--------------------------------------                           -----------------------------------------"<<endl;
        cout<<"-------------------------------------- Processing calls in queue -----------------------------------------"<<endl;
        topo.free_serviced_call_slots(stopping_point+0.5,no_of_calls); // process reaming calls
        cout<<"-------------------------------------------------        -------------------------------------------------"<<endl;
        cout<<"------------------------------------------------- Result -------------------------------------------------"<<endl;
        cout<<"-------------------------------------------------        -------------------------------------------------"<<endl;

        double blocking_probability = ((double)calls_blocked/no_of_calls) ;
        double blocking_probability_new = ((double)calls_blocked_new/no_of_calls) ;

        //Calculate blocking probability //SEE WHY TO USE PRECISION WITH Cout on internet
        cout.precision(4);
        cout<<"FF: Blocking probability is : "<<blocking_probability<<endl;
        cout<<"FF: Normalized contiguous available slots(avg): "<<(vectorSum(ff_cont_slots))/no_of_calls<<endl;
        cout<<"FLF:  Blocking probability is : "<<blocking_probability_new<<endl;
        cout<<"FLF: Normalized contiguous available slots(avg): "<<(vectorSum(flf_cont_slots))/no_of_calls<<endl;

        cout<<"Initial delay: "<<delay_sum/no_of_calls<<endl; // total delay sum / no of calls
        cout<<"Spectrum efficiency(avg): "<<avg_efficiency/no_of_calls<<endl;
        cout<<no_of_calls;

        break;
      }

      if(t_current > Time){
        no_of_calls = i-1;
        stopping_flag = true; // flag to indicate that program should stop after serving all the calls
      }
      else{
        t_hold = topo.next_call_time(mu);

        if(stopping_point < t_current+t_hold)
          stopping_point = t_current+t_hold;

        cout<<endl;
        cout<<"----------------------------------------- AFTER CALL "<<i<<" --------------------------------------------";
        cout<<endl<<"Request arrives at :"<<t_current<<" and call holds for: "<< t_hold <<endl;
        
        topo.call_end_time_new.push_back(t_current + t_hold);

        if( i != 1){
          topo.free_serviced_call_slots(t_current,i); // free slots
        }

        temp_call = topo.call_process(); //returns source, destination, slot required
        vector<int> call(begin(temp_call), end(temp_call));
        call.resize(3);
        //if path doesn't exist decrease no. of calls by 1 and continue
        if(!topo.path_exists(call[1],call[2]))
        {
          i--;
          cout<<"No path exists between node "<<call[1]<<" and node "<<call[2]<<"Hence call isn't valid."<<endl;
          continue;
        }
        
        call_completed = topo.first_fit(call[1],call[2],call[3],i);
        call_completed_new = topo.first_last_fit(call[1],call[2],call[3],i);

         
        if(call_completed != 1)
          calls_blocked++;
        else
          ff_cont_slots.push_back(call_completed);

        if(call_completed_new != 1)
          calls_blocked_new++;
        else
          flf_cont_slots.push_back(call_completed_new);

        cout<<"FF:  Calls arrived : "<<i<<", "<<"Blocked calls : "<<calls_blocked<<endl;
        cout<<"FLF:  Calls arrived : "<<i<<", "<<"Blocked calls : "<<calls_blocked_new<<endl;

        cout.precision(3);
        delay_sum += t_hold;
        avg_efficiency += temp_call[4];
        cout<<"Spectrum efficiency: "<<temp_call[4]<<endl;
      }
    }   
}

