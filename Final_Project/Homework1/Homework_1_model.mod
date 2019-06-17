/*********************************************
 * OPL 12.8.0.0 Model
 * Author: Nicolò Ruggeri
 * Creation Date: 12/dic/2018 at 21:41:26
 *********************************************/

 // Set Time Constraint for Solver (Optional)
execute{

    cplex.tilim = 1000;

}

// Parameters
int N =...;                     // Number of nodes of the graph. Starting node 0, nodes from 0 to N-1
setof(int) V = asSet(0..N-1);   // Vertices of the graph
float C[V][V] = ...;            // Matrix of costs

// Decision Variables
dvar int+ x[V][V];
dvar boolean y[V][V];


// Formulation of the problem 
minimize sum(i in V, j in V) C[i][j]*y[i][j];

subject to{

	InitialFlow: sum(j in V) x[0][j] ==N;
	
	forall(k in V: k!=0){
		FlowPerVertex: sum(i in V)(x[i][k]) - sum(j in V)(x[k][j]) ==1; 	
	}
	
	forall(i in V){
		IncomingEdges: sum(j in V) y[i][j] ==1;	
	}
	
	forall(j in V){
		OutgoingEdges: sum(i in V) y[j][i] ==1;	
	}
	
	forall(i in V, j in V){
		BooleanBound: x[i][j]<=N*y[i][j];	
	}
	
}






