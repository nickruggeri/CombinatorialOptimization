/*********************************************
 * OPL 12.6.1.0 Model
 * Author: luigi
 * Creation Date: Oct 11, 2017 at 8:31:31 PM
 *********************************************/

int Ndest = ...;
setof(string) I = ...;		// closing sites
setof(int) J = asSet(1..Ndest);			// new sites

float	C[I][J] = ...;		// unit transportation cost
int 	D[I] = ...; 		// number of rods available at origin i in I
int 	R[J] = ...; 		// number of rods required at destintion j in J
float 	F = ...;			// fixed cost per available truck
int		N = ...;			// number of available trucks
float 	L = ...;			// fixed cost to rent a truck
int		K = ...;			// truck capacity

dvar int+ x[I][J];			// how may rods to move from i to j ?
dvar boolean y[I][J];		// use a truck from i to j ?
dvar boolean z;				// rent a truck?

minimize sum (i in I, j in J) (C[i][j] * x[i][j]) + F * sum (i in I, j in J) (y[i][j]) + ( L - F ) * z;

subject to {
forall ( j in J ) {
	request: sum ( i in I ) x[i][j] >= R[j];
}

forall ( i in I ) {
	capacity: sum ( j in J ) x[i][j] <= D[i];
}

forall ( i in I, j in J ) {
	useTruck: x[i][j] <= K * y[i][j];
}

limitTrucks: sum (i in I, j in J) (y[i][j]) <= N + z;

}





 
 
