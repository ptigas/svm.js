/*
Copyright (C) 2012 Panagiotis Tigas <ptigas@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, 
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software 
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial 
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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

	this.C = 10.0;

	this.tol = 1e-4;
	this.e = 1e-4;

	this.error_cache = array();

	/* set default kernel to null and
	 * is_linear to true.
	 * If we set the kernel to a different function
	 * then turn is_linear to false.
	 */
	this.kernel = function(a, b) { return dot(a, b); };
	this.is_linear = false;
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
	var loop = 0;

	while ( num_changed > 0 || examine_all ) {
		num_changed = 0;
		if ( examine_all ) {
			//console.log("Examine all")
			// check all training examples
			for ( var i=0; i<this.n; i++ ) {
				num_changed += this.examine_example(i);
			}			
		} else {
			// check only non-boundary cases
			for ( var i=0; i<this.n; i++ ) {
				if ( this.a[i] != 0 && this.a[i] != this.C ) {
					num_changed += this.examine_example(i);
				}
			}			
		}
		//console.log( examine_all + " " + num_changed );
		if ( examine_all == true ) {
			examine_all = false;
		} else if ( num_changed == 0 ) {
			examine_all = true
		}

		if (!( num_changed > 0 || examine_all )) {
			for ( var i=0; i<this.n; i++ ) {
				if ( !this.TTK(i) ) {
					num_changed += this.examine_example(i);
					examine_all = true;
					break;
				}				
			}
		}

		if ( loop++ > 1000 ) {
			break;
		}
		//console.log( num_changed + " " + examine_all );
	}
	
	for ( var i=0; i<this.n; i++ ) {
		console.log( "i=" + i + " ttk=" + this.TTK(i) );	
	}

	console.log( "loops=" + loop );
};

svm.prototype.TTK = function( i ) {
	var alph1 = this.a[i];
	var x1 = this.x[i];
	var y1 = this.y[i];
	var u1 = this.evaluate( x1 );
	var E1 = u1 - y1;

	var r1 = y1 * E1;

	var res = 0;

	if ( ( r1 < -this.e && alph1 < this.C) || (r1 > this.e && alph1 > 0) ) {
		return false;
	} else {
		return true;
	}
}

svm.prototype.examine_example = function( i1 ) {
	var i2 = i1;	
	var alph1 = this.a[i1];
	var x1 = this.x[i1];
	var y1 = this.y[i1];
	var u1 = this.evaluate( x1 );
	var E1 = u1 - y1;

	var r1 = y1 * E1;

	var res = 0;

	if ( ( r1 < -this.e && alph1 < this.C) || (r1 > this.e && alph1 > 0) ) {
		var k=0, i2=i1;
		var tmax=0;

		while ( i2 === i1 ) i2 = Math.floor(Math.random()*this.n;

		return this.solve_lagrange( i1, i2 );

		/*
		for ( i2 = (-1), tmax = 0, k = 0; k < this.n; k++) {
			if ( this.a[k] > 0 && this.a[k] < this.C ) {
				var E2=0, temp=0;
				console.log( "ASDFAF" );

				E2 = this.evaluate(this.x[i2]) - this.y[i2];
				temp = Math.abs(E1 - E2);
				if (temp > tmax) {
					tmax = temp;
					i2 = k;
				}
			}

			if ( i2 >= 0 ) {
				if ( this.solve_lagrange( i1, i2 ) == 1) {
					return 1;
				}
			}
		}
		*/
	}	
		
	//console.log("i1="+i1+" i2="+i2+" a="+this.a + " res=" + res );
	return res;
}

svm.prototype.compute_weights = function() {
	var w= new Array(2);
      for(var j=0;j<2;j++) {
        var s= 0.0;
        for(var i=0;i<this.n;i++) {
          s+= this.a[i] * this.y[i] * this.x[i][j];
        }
        w[j]= s;
      }
      return {w: w, b: this.b};
}

svm.prototype.solve_lagrange = function( i1, i2 ) {
	//console.log("ASDFASFAS");
	var a1 = this.a[i1];
	var a2 = this.a[i2];
	var x1 = this.x[i1];
	var x2 = this.x[i2];
	var y1 = this.y[i1];
	var y2 = this.y[i2];	
	var u1 = this.evaluate(x1);
	var u2 = this.evaluate(x2);

	if ( i1 === i2 ) {
		throw "i1 and i2 are equal."
	}

	var a1n = 0.0;
	var a2n = 0.0;

	var s = y1 * y2;

	var E1 = u1 - y1;
	var E2 = u2 - y2;

	if ( y1 != y2 ) {
		var L = Math.max( 0, a2-a1 );
		var H = Math.min( this.C, this.C+a2-a1 );	
	} else {
		var L = Math.max( 0, a1+a2-this.C );
		var H = Math.min( this.C, a1+a2 );
	}

	if ( Math.abs( L - H ) < this.e ) {
		return 0;
	}

	if ( this.is_linear ) {
		var K = function(a, b) { return dot(a, b); };
	} else {
		var K = this.kernel;	
	}

	var k11 = K(x1, x1);
	var k22 = K(x2, x2);
	var k12 = K(x1, x2);
	
	h = k11 + k22 - 2*k12;

	if ( h > 0 ) {		
		var a2n = a2 + ( y2 * ( E1 - E2 ) ) / h;

		if ( a2n < L ) {
			a2n = L;			
		} else if ( a2n > H ) {
			a2n = H;
		}

	} else {
		return 0
		/* h is not positive - computing Ψ */
		var f1 = y1*(E1+this.b) - a1*k11 - s*a2*k12;
		var f2 = y2*(E2+this.b) - s*a1*k12 - a2*k22;
		var L1 = a1 + s*(a2-L);
		var L2 = a1 + s*(a2-L);

		var psiL = L1*f1 + L*f2 + .5*L1*L1*k11 + .5*L*L*k22 + S*L*L1*k12;
		var psiH = H1*f1 + H*f2 + .5*H1*H1*k11 + .5*H*H*k22 + S*H*H1*k12;
		
		if ( psiL < psiH - this.e ) {
			var a2n = L;			
		} else if ( psiL > psiH + this.e ) {
			var a2n = H;
		} else {
			var a2n = a2;
		}
	}
	//var a2nc = Math.min( H, Math.max(a2n, L) ); // clipt a2n
	//a2n = a2nc;
	var a2nc = a2n;
	if ( a2nc > H ) a2nc = H;
	if ( a2nc < L ) a2nc = L;
	//var a2nc = a2n;

	if ( Math.abs(a2n-a2) < this.e*(a2+a2n+this.e) ) {
		return 0;
	}
	a1n = a1+s*(a2-a2n);

	/* update the threshold b */
	var b1 = E1 + y1 * ( a1n - a1 ) * k11 + y2 * ( a2n - a2 ) * k12 + this.b;
	var b2 = E2 + y1 * ( a1n - a1 ) * k11 + y2 * ( a2n - a2 ) * k22 + this.b;
	this.b = (b1+b2)/2.0;
	//console.log( "bs= " + b1 + " " + b2 );
	console.log( "as= " + a1n + " " + a2n );
	//if(a1n > 0 && a1n < this.C) this.b= b2;
    //if(a2n > 0 && a2n < this.C) this.b= b1;

	//console.log( "b="+this.b  );


	if (a1 < 0) {
		a2n += s*a1n;
		a1n = 0;
	} else if (a1.n > this.C) {
		a2n += s * (s1n-this.C);
		a1n = C;
	}
	

	a1 = a1n;
	a2 = a2n;

	this.a[i1] = a1;
	this.a[i2] = a2;

	/* update error cache */
	this.error_cache[ i1 ] = this.evaluate( x1 ) - this.y[i1]; 
	this.error_cache[ i2 ] = this.evaluate( x2 ) - this.y[i2];

	return 1;
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
