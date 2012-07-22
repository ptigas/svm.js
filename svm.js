#!/usr/bin/env node

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

	this.C = 1.0;

	this.tol = 1e-4;
	this.e = 1e-4;

	/* set default kernel to null and
	 * is_linear to true.
	 * If we set the kernel to a different function
	 * then turn is_linear to false.
	 */
	this.kernel = function(a, b) { return dot(a, b); };
	this.is_linear = false;

	console.log( "C: " + C );
}

svm.prototype.train = function( xs, ys ) {
	if ( xs.length != ys.length ) {
		throw "Xs are not as many as Ys";
	}

	// set n, x, y and a arrays.
	this.n = xs.length;
	this.x = xs;
	this.y = ys;
	

	this.a = new Array(this.n);
	for ( var i=0; i<this.n; i++ ) {
		this.a[i] = 0.0;
	}

	this.b = 0.0;
	
	var num_changed = 0;
	var examine_all = true;
	while ( num_changed > 0 || examine_all ) {
		num_changed = 0;
		if ( examine_all ) {
			console.log("Examine all")
			// check all training examples
			for ( var i=0; i<this.n; i++ ) {
				num_changed += (this.examine_example(i)?1:0);
			}
			examine_all = false;
		} else {
			// check only non-boundary cases
			for ( var i=0; i<this.n; i++ ) {
				if ( this.a[i] != 0 && this.a[i] != this.C ) {
					num_changed += (this.examine_example(i)?1:0);
				}
			}
			if ( num_changed == 0 ) {
				examine_all = true;	
			}
		}
		console.log( num_changed + " " + examine_all );
	}
};

svm.prototype.examine_example = function( i2 ) {
	var i1 = i2;
	while ( i1 === i2 ) {
		i1 = Math.floor( Math.random()*this.n );
	}
	var res = this.solve_lagrange( i1, i2 );
	console.log("i1="+i1+" i2="+i2+" a="+this.a + " res=" + res );
	return res;
}

svm.prototype.solve_lagrange = function( i1, i2 ) {
	var a1 = this.a[i1];
	var a2 = this.a[i2];
	var x1 = this.x[i1];
	var x2 = this.x[i2];
	var y1 = this.y[i1];
	var y2 = this.y[i2];	
	var u1 = this.evaluate(x1);
	var u2 = this.evaluate(x2);

	var s = y1*y2;

	var E1 = u1 - y1;
	var E2 = u2 - y2;

	if ( y1 != y2 ) {
		var L = Math.max( 0, a2-a1 );
		var H = Math.min( this.C, this.C+a2-a1 );	
	} else {
		var L = Math.max( 0, a2+a1-C );
		var H = Math.min( C, a2+a1 );
	}

	if ( L == H ) {
		return false;
	}

	if ( this.is_linear ) {
		var K = function(a, b) { return dot(a, b); };
	} else {
		var K = this.kernel;	
	}	
	
	h = K(x1, x1) + K(x2, x2) - 2*K(x1, x2);

	if ( h > 0 ) {		
		var a2n = a2 + ( y2*( E1 - E2 ) ) / h;		
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
			var a2n = a2;
		}
	}
	var a2nc = Math.min( H, Math.max(a2n, L) ); // clipt a2n
	
	if ( Math.abs(a2n-a2) < this.e*(a2+a2n+this.e) ) {
		return false;
	}
	a1n = a1+s*(a2-a2n);

	/* update the threshold b */
	var b1 = E1+y1*(a1n+a1)*K(x1,x1)+y2*(a2nc-a2)*K(x1,x2)+this.b;
	var b2 = E2+y1*(a1n+a1)*K(x1,x1)+y2*(a2nc-a2)*K(x2,x2)+this.b;
	this.b = (b1+b2)/2.0;

	//console.log( "E="+this.b  );

	a1 = a1n;
	a2 = a2n;

	this.a[i1] = a1;
	this.a[i2] = a2;

	return true;
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
			//console.log( "evaluation x="+x+" ---> s="+this.b);
		}
		u = s - this.b;
	}

	//console.log( "evaluation x="+x+" ---> "+u );

	return u;
}

var x = [
	[0,0],
	[1,1],
];

var y = [
	-1,
	1
];

s = new svm( 1.0 );

s.train( x, y )

console.log( x );

console.log( s.a );