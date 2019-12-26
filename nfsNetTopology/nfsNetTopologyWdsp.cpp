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
#include<numeric>
#include <tuple>

using namespace std;

//define a weighted edge as pair of destination and weight(directed)
typedef pair<int,float> Edge ;
//define adjacency list as a vector of vector of edges so as to access particular index as well as add/remove nodes at runtime
typedef vector< vector< Edge > >  AdjacencyList;

typedef pair< int, pair<float,bool> > sl_Edge;
typedef vector< vector< sl_Edge > >  sl_AdjacencyList;

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

        vector< vector <sl_AdjacencyList> > slot_arrays;//slot arrays for each channel (wavelength) and link
        vector< vector <sl_AdjacencyList> > wdsp_slot_arrays;
        //-----------------------------------------------------------------------------------------------------//
        //vector< vector <int> > traffic_matrix;//traffic matrix showing call requests from source to destination; USING VECTORS AS UNABLE TO USE THIS->NODES
        vector<double> call_end_time;//vectors to store end time for call requests
        vector<double> wdsp_call_end_time;

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
        //Create empty slots in each link and channel
        void create_empty_slots();
        //print slot matrix
        void print_slot_matrix();

        //simulate call arrival process
        vector<float> call_process();
        //generate time after which next call occurs
        float next_call_time(double lambda);
        // get no of slots available
        void slots_available(vector<Edge> path,int wl,bool slot_available[],bool fresh_slots[],bool wdsp);
        // get wavelength according to first-fit algorithm
        float first_fit(int source,int destination,int slots_required,int request_number);
        //wdsp algorithm
        float wdsp(int source,int destination,int slots_required,int request_number,float t_hold);
        
        //function to deallocate slots for call which has been serviced
        void free_serviced_call_slots(double time,int calls_till_now);
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
  //adjList[destination].push_back(make_pair(source,weight)); //duplex network
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

//Create empty slot arrays
void WDM::create_empty_slots()
{
    int r,q,i,j;

    slot_arrays.clear(); // first fit slots
    slot_arrays.resize(this->num_slots);//set size of slot_arrays

    wdsp_slot_arrays.clear(); // wdsp slots
    wdsp_slot_arrays.resize(this->num_slots);

    for(r=0;r<slot_arrays.size();r++)
    {
        slot_arrays[r].resize(num_wavelengths);//resize all slot arrays to number of wavelengths
        wdsp_slot_arrays[r].resize(num_wavelengths);

        for(q=0;q<num_wavelengths;q++)//loop till number of wavelegths
        {
            slot_arrays[r][q].resize(this->nodes);//set size of wavelegth q matrix to number of nodes
            wdsp_slot_arrays[r][q].resize(this->nodes);

            for(i=0;i<this->nodes;i++)//inner loop till number of nodes
            {
                slot_arrays[r][q][i].resize(Network[i].size());//set size of wavelegth q matrix for node i to number of links
                wdsp_slot_arrays[r][q][i].resize(Network[i].size());
                {
                for(j=0;j<Network[i].size();j++)//second inner loop till number of edges from that node
                    {
                        //------
                        slot_arrays[r][q][i][j].first = Network[i][j].first;//initialize destination node from topology
                        slot_arrays[r][q][i][j].second.first = 0;//intialize qth wavelength as 0(free)
                        slot_arrays[r][q][i][j].second.second = false;
                        //-------

                        // slot_arrays[r][q][i][j].first = Network[i][j].first;//initialize destination node from topology
                        // slot_arrays[r][q][i][j].second = 0;//intialize qth wavelength as 0(free)

                        wdsp_slot_arrays[r][q][i][j].first = Network[i][j].first;
                        wdsp_slot_arrays[r][q][i][j].second.first = 0;
                        wdsp_slot_arrays[r][q][i][j].second.second = false;
                    }
                }
            }
        }
    }
}

void WDM::print_slot_matrix()
{
  int r,q,i,j;

  for(r=0;r<slot_arrays.size();r++){
    for(q=0;q<num_wavelengths;q++)
      for(i=0;i<this->nodes;i++){//inner loop till number of nodes
        for(j=0;j<Network[i].size();j++)//second inner loop till number of edges from that node
          cout<<slot_arrays[r][q][i][j].second.first<<",";
        cout<<endl;
      }
    cout<<"-------"<<endl;
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

  result[1]=source;
  result[2]=destination;
  result[3]=slots_required;
  result[4]=(float)bandwidth_required/(float)(slots_required*this->slot_width); // spectrum efficiency

  cout.precision(5);
  cout<<"Request between Source node "<<result[1]<<" & Destination node "<<result[2]<<" for bandwidth "<<bandwidth_required\
  <<" GHz, "<<"no. of slots required " <<result[3]<<endl<<endl;
  
  return result;
}

//generate time after which next call occurs
float WDM::next_call_time(double service_rate)
{return -logf(1.0f - (float) random() / (RAND_MAX + 1)) / service_rate;}


void WDM::slots_available(vector<Edge> path,int wl,bool slot_available[],bool fresh_slots[],bool wdsp = false)
{
  int i,j,origin,end,slot_number;

  for(i=0;i<num_slots;i++){
      slot_available[i] = true;
      fresh_slots[i] = true;
  }

  if(wdsp)
    for(i=path.size()-1;i>0;i--){//inner loop till number of nodes on path  
      origin = path[i].first;
      end = path[i-1].first;

      for(slot_number=0;slot_number<num_slots;slot_number++){//second inner loop till number of slots in wavelength
        for(j=0;j<Network[origin].size();j++)// to get the destination node
          if (wdsp_slot_arrays[slot_number][wl][origin][j].first == end)
            break;
        if(wdsp_slot_arrays[slot_number][wl][origin][j].second.first==0)
          slot_available[slot_number]=(slot_available[slot_number]&&true);
        else
          slot_available[slot_number]=false;

        if(wdsp_slot_arrays[slot_number][wl][origin][j].second.second==false)
          fresh_slots[slot_number]=(fresh_slots[slot_number]&&true);
        else
          fresh_slots[slot_number]=false;
      }
    }
  else
    for(i=path.size()-1;i>0;i--){//inner loop till number of nodes on path
      origin = path[i].first;
      end = path[i-1].first;

      for(slot_number=0;slot_number<num_slots;slot_number++){//second inner loop till number of slots in wavelength
        for(j=0;j<Network[origin].size();j++)// to get the destination node
          if (slot_arrays[slot_number][wl][origin][j].first == end)
            break;
        if(slot_arrays[slot_number][wl][origin][j].second.first==0)
          slot_available[slot_number]=(slot_available[slot_number]&&true);
        else
          slot_available[slot_number]=false;

        if(slot_arrays[slot_number][wl][origin][j].second.second==false)
          fresh_slots[slot_number]=(fresh_slots[slot_number]&&true);
        else
          fresh_slots[slot_number]=false;
      }
    }
}

//get wavelength according to first-fit algorithm
float WDM::first_fit(int source,int destination,int slots_required,int request_number)
{
  int i,j,origin,end,q,slot_number;
  int cfs,ffs = 0;//variable to represent continuous number of free slots
  int last,start;//variable to represent starting & last free slot number
  bool wavelength_available;//boolean to represent wavelength availability on this path
  bool slot_available[num_slots];//boolean array to represent slot availability across path & wavelength
  bool fresh_slots[num_slots];

  vector<Edge> path = this->Dijkstra(source,destination);

  //cout<<"Shortest path between source node "<<source<<" & destination node "<<destination<<" is"<<endl;
  cout<<endl<<endl<<"PATH: ";
  for(i=path.size()-1;i>0;i--)//print the shortest path
      cout<<path[i].first<<"->"<<path[i-1].first<<"("<<path[i].second<<")"<<" ";

  cout<<endl;

  for(q=0;q<this->num_wavelengths;q++)//loop till number of wavelengths
  {
    start=0;
    cfs=0;

    wavelength_available = true; // or link

    slots_available(path,q,slot_available,fresh_slots); //update available and fresh slots in the link
    cout<<endl;
    
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
      cout<<"Requested capacity(Qr), not available."<<endl<<endl;
      continue;
    }
    
    if(wavelength_available)
    {
      last = start + slots_required -1 ;
      cout<<endl<<"FF: slots alloted from "<<start<<" to "<<last<<endl;
      for(i=path.size()-1;i>0;i--)//inner loop till number of nodes on path
      {    
        origin = path[i].first;
        end = path[i-1].first;

        for(slot_number=start;slot_number<=last;slot_number++)//second inner loop till number of slots in wavelength
            {   
                for(j=0;j<Network[origin].size();j++)//Only to find the number of edge which contains destination
                    if (slot_arrays[slot_number][q][origin][j].first == end)
                      break;

                this->slot_arrays[slot_number][q][origin][j].second.first = request_number;//Mark the slots with the call request it is alloted to
                this->slot_arrays[slot_number][q][origin][j].second.second = true; // slot used                   
            }  
      }
      slots_available(path,q,slot_available,fresh_slots);
      int temp = 0;
      for(i=0;i<num_slots;i++){
        if(fresh_slots[i])
          temp++;
        else{
          ffs = max(ffs,temp);
          temp = 0;}
      }
      ffs = max(ffs,temp);
      cout<<"Max unused-contiguous slots: "<<ffs<<endl;
      return (float)ffs/(num_slots*path.size());
    }
  }
  return 9;
}


float WDM::wdsp(int source,int destination,int slots_required,int request_number,float t_hold)
{
  int i,j,origin,end,q,slot_number;
  int p,s,temp_slot_number;
  int cfs,ffs=0;//variable to represent continuous number of free slots
  int last,start;//variable to represent starting & last free slot number
  bool wavelength_available;//boolean to represent wavelength availability on this path
  bool slot_available[num_slots];//boolean array to represent slot availability across path & wavelength
  bool fresh_slots[num_slots];

  vector<Edge> path = this->Dijkstra(source,destination);
  cout<<endl;

  for(q=0;q<this->num_wavelengths;q++)//loop till number of wavelengths
  {
    start=0;
    cfs=0;

    wavelength_available = true;
    // available slots
    slots_available(path,q,slot_available,fresh_slots,true); //update available and fresh slots in the link
    cout<<endl;

    float Nr = t_hold * slots_required;

    if(Nr <= 2){ // lower index slots
      for(i=0;i<num_slots;i++){
        if(slot_available[i])
          cfs++;
        else{
          start=i+1; cfs=0;
        }
        if(cfs==slots_required)
          break;
      }
    }
    else{ // larger index slots
      for(i=num_slots-1;i>=0;i--){
        if(slot_available[i])
          cfs++;
        else
          cfs=0;
        if(cfs==slots_required){
          start=i;
          break;
        }
      }
    }
    if(cfs<slots_required)
      {
        wavelength_available=false;
        cout<<"Requested capacity(Qr), not available."<<endl<<endl;
        continue;
      }
    
    if(wavelength_available)
    {
      last = start + slots_required -1;
      cout<<"WDSP: slots alloted from "<<start<<" to "<<last<<endl;
      for(i=path.size()-1;i>0;i--)//inner loop till number of nodes on path
        {    
          origin = path[i].first;
          end = path[i-1].first;

          for(slot_number=start;slot_number<=last;slot_number++)//second inner loop till number of slots in wavelength
              {   
                for(j=0;j<Network[origin].size();j++)//USELESS LOOP; Only to find the number of edge which contains destination
                  if (wdsp_slot_arrays[slot_number][q][origin][j].first == end)
                    break;
                this->wdsp_slot_arrays[slot_number][q][origin][j].second.first = request_number;//Mark the slots with the call request it is alloted to                     
                this->wdsp_slot_arrays[slot_number][q][origin][j].second.second = true;
              }  
        }
      slots_available(path,q,slot_available,fresh_slots,true);
      int temp = 0;
      for(i=0;i<num_slots;i++){
        if(fresh_slots[i])
          temp++;
        else{
          ffs = max(ffs,temp);
          temp = 0;}
      }
      ffs = max(ffs,temp);
      cout<<"Max unused-contiguous slots: "<<ffs<<endl;
      return (float)ffs/(num_slots*path.size());
    }
  }
  return 9;
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
                          if(slot_arrays[r][q][i][j].second.first == call_number)//if slot has been alloted to that call number
                              slot_arrays[r][q][i][j].second.first = 0;//free that slot
        call_end_time[call_number] = -1;//mark call as having been serviced
        cout<<"FF::Call "<<call_number<<", completed."<<endl;
      }
  }

  cout<<"FF::Calls completed : "<<calls_completed<<", Calls in progress : "<<calls_in_progress<<endl;

  // free WDSP calls  
  calls_completed=0,calls_in_progress=0;

  for(call_number=1;call_number<wdsp_call_end_time.size();call_number++)
  {
    if(wdsp_call_end_time[call_number]==-1)//call has been serviced
      calls_completed++;
    else if(wdsp_call_end_time[call_number]>time)//If the call end time for ith request is after current time then continue
      calls_in_progress++;
    else
      {
        for(r=0;r<wdsp_slot_arrays.size();r++)//loop till number of slots
          for(q=0;q<num_wavelengths;q++)//loop till number of wavelegths
              for(i=0;i<this->nodes;i++)//inner loop till number of nodes
                    for(j=0;j<Network[i].size();j++)//second inner loop till number of edges from that node
                          if(wdsp_slot_arrays[r][q][i][j].second.first == call_number)//if slot has been alloted to that call number
                              wdsp_slot_arrays[r][q][i][j].second.first = 0;//free that slot
        wdsp_call_end_time[call_number] = -1;//mark call as having been serviced
        cout<<"WDSP::Call "<<call_number<<", completed."<<endl;
      }
  }

  cout<<"WDSP::Calls completed : "<<calls_completed<<", Calls in progress : "<<calls_in_progress<<endl;

}


double vector_sum(vector<float> v){
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

    int no_of_wavelength = 1;
    topo.set_wavelength_number(no_of_wavelength);

    cout<<"Enter number of slots per link:"<<endl;
    cin>>slots; 
    topo.set_slot_number(slots);

    cout<<"Enter width of slot per link:"<<endl;
    cin>>width; 
    topo.set_slot_width(width);
    topo.create_empty_slots();

    //----------------------------------------------------------------------------------------//
   
    cout<<endl<<"Enter call arrival rate (inverse of average duration between calls) lambda:"<<endl;
    cin>>lambda;
    cout<<"Enter service rate mu (inverse of average holding time):"<<endl<<"NOTE: SHOULD BE LESS THAN LAMBDA:"<<endl;
    cin>>mu;
    cout<<"Enter time till when to simulate call arrivals and serivce:"<<endl;
    cin>>Time;

    int calls_blocked,wdsp_calls_blocked,i,total_calls = ceil(lambda*Time);
    float call_completed,wdsp_call_completed;
    //-----------------------------------------------------------------------------------//
    double t_current,t_hold;
    topo.call_end_time.clear(); //vector to store end time for call requests
    topo.wdsp_call_end_time.clear();
    
    srand(time(NULL));
    calls_blocked = wdsp_calls_blocked = 0;

    topo.call_end_time.push_back(-1);
    topo.wdsp_call_end_time.push_back(-1);

    //---------------------------------------------------------------------------------------//
    vector<float> temp_call;
    temp_call.resize(4);

    float delay_sum = 0;
    float avg_efficiency = 0;
    float stoping_point = 0;
    int no_of_calls;
    bool stoping_flag = false;
    vector<float> cont_slots;
    vector<float> wdsp_cont_slots;
    //---------------------------------------------------------------------------------------//

    for(i=1;;i++)
    {
      
      t_current = t_current - topo.next_call_time(lambda);

      if(stoping_flag){ // current time is greater than last serviced call time
        cout<<"--------------------------------------                           -----------------------------------------"<<endl;
        cout<<"-------------------------------------- Processing calls in queue -----------------------------------------"<<endl;
        topo.free_serviced_call_slots(stoping_point+0.5,no_of_calls); // process reaming calls
        cout<<"-------------------------------------------------        -------------------------------------------------"<<endl;
        cout<<"------------------------------------------------- Result -------------------------------------------------"<<endl;
        cout<<"-------------------------------------------------        -------------------------------------------------"<<endl;

        //Calculate blocking probability //SEE WHY TO USE PRECISION WITH Cout on internet
        cout.precision(4);
        cout<<endl;
        cout<<"FF: Blocking probability is : "<<((double)calls_blocked/no_of_calls)<<endl;
        cout<<"FF: Normalized contiguous available slots(avg): "<<vector_sum(cont_slots)/no_of_calls<<endl;
        cout<<endl;
        cout<<"WDSP:  Blocking probability is : "<<((double)wdsp_calls_blocked/no_of_calls)<<endl;
        cout<<"WDSP: Normalized contiguous available slots(avg): "<<vector_sum(wdsp_cont_slots)/no_of_calls<<endl;
        cout<<endl;

        cout<<"Initial delay: "<<delay_sum/no_of_calls<<endl; // total delay sum / no of calls
        cout<<"Spectrum efficiency(avg): "<<avg_efficiency/no_of_calls<<endl<<endl;
        break;
      }

      if(t_current > Time){
        no_of_calls = i-1;
        stoping_flag = true; // flag to indicate that program should stop after serving all the calls
      }
      else{
        t_hold = -topo.next_call_time(mu);

        if(stoping_point < t_current+t_hold)
          stoping_point = t_current+t_hold;

        cout<<endl;
        cout<<"----------------------------------------- AFTER CALL "<<i<<" --------------------------------------------"<<endl;
        cout<<"Request arrives at :"<<t_current<<" and call holds for: "<< t_hold <<endl;
        
        topo.call_end_time.push_back(t_current + t_hold);
        topo.wdsp_call_end_time.push_back(t_current + t_hold);

        temp_call = topo.call_process(); //returns source, destination, slot required
        vector<int> call(begin(temp_call), end(temp_call));
        call.resize(3);

        if( i != 1){
          topo.free_serviced_call_slots(t_current,i); // free slots
        }
        //if path doesn't exist decrease no. of calls by 1 and continue
        if(!topo.path_exists(call[1],call[2]))
        {
          i--;
          cout<<"No path exists between node "<<call[1]<<" and node "<<call[2]<<"Hence call isn't valid."<<endl;
          continue;
        }
        
        call_completed = topo.first_fit(call[1],call[2],call[3],i);
        if(call_completed <=1){
          cout<<"Normalized contiguous available slots: "<<call_completed;
          cont_slots.push_back(call_completed);
        }
        else
          calls_blocked++;

        wdsp_call_completed = topo.wdsp(call[1],call[2],call[3],i,t_hold);
        if(wdsp_call_completed <=1){
          cout<<"Normalized contiguous available slots: "<<wdsp_call_completed<<endl<<endl;
          wdsp_cont_slots.push_back(wdsp_call_completed);
        }
        else
          wdsp_calls_blocked++;

        cout<<"FF:  Calls arrived : "<<i<<", "<<"Blocked calls : "<<calls_blocked<<endl;
        cout<<"WDSP:  Calls arrived : "<<i<<", "<<"Blocked calls : "<<wdsp_calls_blocked<<endl;

        cout.precision(3);
        delay_sum += t_hold;
        avg_efficiency += temp_call[4];
        cout<<"Spectrum efficiency: "<<temp_call[4]<<endl;
      }
    }   
}

