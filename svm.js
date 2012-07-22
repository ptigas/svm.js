
function dot( A, B ) {
	if ( A.length != B.length ) {
		throw "Different length"
	}
	var s = 0.0;
	for ( var i=0; i<A.length; i++ ) {
		s += A[i]*B[i];
	}
	return s;
}

function svm( C ) {
	this.version = "0.0";	

	// Initialize to null.
	this.a = null;
	this.w = null;
	this.x = null;
	this.y = null;
	this.n = null;
	this.b = null;

	/* set default kernel to null and
	 * is_linear to true.
	 * If we set the kernel to a different function
	 * then turn is_linear to false.
	 */
	this.kernel = null;
	this.is_linear = true;

	console.log( "C: " + C );
}

svm.prototype.train = function( xs, ys ) {

	if ( xs.length != ys.length ) {
		throw "Xs are not as many as Ys";
	}

	// set n, x, y and a arrays.
	this.n = xs.length;
	this.x = xs
	this.y = ys
	this.a = []
	
	console.log( "n: " + xs.length );	
};

/**
 * Function to check if 
 * Karush-Kuhn-Tucker (KKT) conditions
 * are valid.
 *
 * @param i number of example to test
 * @returns boolean
 */
svm.prototype.kkt = function( i ) {
	if ( this.a === null ) {
		throw "alphas are null.";
	}

	if ( this.x === null ) {
		throw "xs are null.";
	}

	if ( this.y === null ) {
		throw "xs are null.";
	}

	if ( this.x.length != this.y.length != this.a.length != this.n ) {
		throw "xs, ys and alphas don't have the same length."
	}

	/* KKT conditions
	 *
	 * a_i = 0 	   <=> y_i*u_i >= 1
	 * 0 < a_i < C <=> y_i*u_i == 1
	 * a_i = C     <=> y_i*u_i <= 1
	 */
	var u = this.evaluate( this.x[i] );
	if ( this.a[i] == 0 ) {		
		if ( ( this.y[i] * u ) >= 1 ) {
			return true;
		}
	} else if ( this.a[i] == this.C ) {
		if ( ( this.y[i] * u ) == 1 ) {
			return true;
		}
	} else {
		if ( ( this.y[i] * u ) <= 1 ) {
			return true;
		}
	}
	return false;
}

svm.prototype.evaluate = function( x ) {
	var u = 0.0;
	if ( this.is_linear == true ) {
		/* use w to evaluate svm */
		u = dot( this.w, this.x ) - b;

	} else {
		/* use kernel to evaluate svm */
		if ( this.kernel == null ) {
			throw "is_lenear is false and kernel is not defined."
		}
		var s = 0.0;
		for ( var i=0; i<this.n; i++ ) {
			s += this.y[i]*
				 this.a[i]*
				 this.kernel( this.x[i], x );
		}
		u = s - b;
	}
	return u;
}

var x = [
	[1,2],
	[3,4],
];

var y = [
	1,
	-1
];

s = new svm( 1.0 );

s.train( x, y )

console.log( x );