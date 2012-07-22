/*
Copyright (C) 2012 Panagiotis Tigas (ptigas@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

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

	this.tol = 1e-4;

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
	
	var num_changed = 0;
	var examine_all = true;
	while ( num_changed > 0 || examine_all ) {
		num_changed = 0;
		if ( examine_all ) {
			// check all training examples
			if ( this.a[i] != 0 && this.a[i] != C ) {
				num_changed += this.examine_example(i);
			}
		} else {
			// check only non-boundary cases
			for ( var i=0; i<this.n; i++ ) {
				if ( this.a[i] != 0 && this.a[i] != C ) {
					num_changed += this.examine_example(i);
				}
			}
			if ( num_changed == 0 ) {
				examine_all = true;	
			}
		}		
	}
};

svm.prototype.solve_lagrange = function( i1, i2 ) {
	var a1 = this.a[i1];
	var a2 = this.a[i2];
	var x1 = this.x[i1];
	var x2 = this.x[i2];
	var y1 = this.y[i1];
	var y2 = this.y[i2];	
	var u1 = this.evaluate(x1);
	var u2 = this.evaluate(x2);

	var E1 = u1 - y1;
	var E2 = u2 - y2;

	if ( y1 != y2 ) {
		var L = Math.max( 0, a2-a1 );
		var H = Math.min( this.C, this.c+a2-a1 );	
	} else {
		var L = Math.max( 0, a2+a1-C );
		var H = Math.min( C, a2+a1 );
	}

	if ( this.is_linear ) {
		var K = function(a, b) { return dot(a, b); };
	} else {
		var K = this.kernel;	
	}	
	
	h = K(x1, x1) + K(x2, x2) - 2*K(x1, x2);

	if ( h > 0 ) {		
		var a2n = a2 + ( y2*( E1 - E2 ) ) / h;
		var a2n = Math.min( H, Math.max(a2nc, L) ); // clipt a2n
	} else {
		/* h is not positive - computing Ψ */
		var f1 = y1*(E1+this.b) - a1*K(x1,x1) - s*a2*K(x1,x2);
		var f2 = y2*(E2+this.b) - s*a1*K(x1,x2) - a2*K(x2,x2);
		var L1 = a1 + s*(a2-L);
		var L2 = a1 + s*(a2-L);
		var psiL = L1*f1 + L*f2 + .5*L1*L1*K(x1,x1) + .5*L*L*K(x2,x2) + S*L*L1*K(x1,x2);
		var psiH = H1*f1 + H*f2 + .5*H1*H1*K(x1,x1) + .5*H*H*K(x2,x2) + S*H*H1*K(x1,x2);
		if ( psiL < psiH - this.e ) {
			var a2n = L;			
		} else if ( psiL > psiH + this.e ) {
			var a2n = H;
		} else {
			var a2 = a2;
		}
	}
	
	if ( Math.abs(a2n-a2) < this.e*(a2+a2n+this.e) ) {
		return false;
	}
	a1 = a1+s*(a2-a2n);
	return true;
}

svm.prototype.examine_example = function( i2 ) {
	x2 = this.x[i2];
	y2 = this.evaluate.( x2 );
	u2 = this.y[i2];
	a2 = this.a[i2];
	E2 = u2-y2;
	r2 = E2*y2;
	if ( (r2 < -this.tol && a2 < C ) ||
		 (r2 >  this.tol && a2 > C ) ) {

	}
}

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