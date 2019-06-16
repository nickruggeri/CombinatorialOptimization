/*********************************************
 * OPL 12.6.1.0 Model
 * Author: luigi
 * Creation Date: Oct 11, 2017 at 8:31:31 PM
 *********************************************/

setof(string) I = ...;		// jobs
setof(string) K = ...;		// machines
setof(int) P = asSet(0..card(K)-1);

float	R[I] = ...;			// release times
float 	D[I][K] = ...; 		// processing times
string 	sigma[I][P] = ...;	// task order
float 	M = max(i in I) R[i] + sum(i in I, k in K)(D[i][k]);

dvar float+		y;			// makespan
dvar float+		h[I][K];	// start time
dvar boolean 	x[i in I][j in I][K];	// precedence

minimize y;

subject to {

forall ( i in I ) {
makespan:	y >= h[i][sigma[i][card(K)-1]] + D[i][sigma[i][card(K)-1]];
release:	h[i][sigma[i][0]] >= R[i]; 
	forall (p in P: p >= 1) { 
	taskSeq:	h[i][sigma[i][p]] >= h[i][sigma[i][p-1]] + D[i][sigma[i][p-1]];
	}
	forall ( j in I, k in K : i != j) {
	machSeq1:	h[i][k] >= h[j][k] + D[j][k] - M * x[i][j][k];
	machSeq2:	h[j][k] >= h[i][k] + D[i][k] - M * (1 - x[i][j][k]);
	}
}
forall ( i in I, j in I, k in K : i == j) notUsedX: x[i][j][k] == 0;	
}





 
 
