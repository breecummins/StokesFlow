#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>


static PyObject *
cubstokes2D_matmult(PyObject *self, PyObject *args)
{
	PyArrayObject *obspts, *nodes, *f, *out;
	double eps, mu, dx, dy, dx2, dy2, r2, H1, H2, G1, G2, G3, val1, val2, ptx, pty, eps2;
	long i, j, ns, np;
	if (!PyArg_ParseTuple(args, "ddOOO", &eps, &mu, &obspts, &nodes, &f))
			return NULL;
	np=2*(obspts->dimensions[0]);
	ns = nodes->dimensions[0];
	out = PyArray_ZEROS(1, &np, NPY_FLOAT64, 0);
	for (j=0;j<np/2;j++) {
		ptx = *(double *)PyArray_GETPTR2(obspts, j, 0);
		pty = *(double *)PyArray_GETPTR2(obspts, j, 1);
		val1 = 0.0;
		val2 = 0.0;
		for (i=0;i<ns;i++) {
			dx = ptx - *(double *)PyArray_GETPTR2(nodes, i, 0);
			dy = pty - *(double *)PyArray_GETPTR2(nodes, i, 1);
			dx2 = dx*dx;
			dy2 = dy*dy;
			eps2 = eps*eps;
			r2 = dx2 + dy2 + eps2;
			H2 = (2/r2)/(8*M_PI*mu);
			H1 = eps2*H2 - log(r2)/(8*M_PI*mu);
			G1 = H1 + dx2*H2;
			G2 = H1 + dy2*H2;
			G3 = dx*dy*H2;
			val1 += G1* *(double *)PyArray_GETPTR2(f, i, 0);
			val1 += G3* *(double *)PyArray_GETPTR2(f, i, 1);
			val2 += G3* *(double *)PyArray_GETPTR2(f, i, 0);
			val2 += G2* *(double *)PyArray_GETPTR2(f, i, 1);

		}
		*(double *)PyArray_GETPTR1(out, 2*j) = val1;
		*(double *)PyArray_GETPTR1(out, 2*j+1) = val2;

	}

	return out;
}


static PyObject *
cubstokes2D_stressDeriv(PyObject *self, PyObject *args)
{
	PyArrayObject *gradub, *gradlt, *F, *P, *Finv, *fmm, *out;
	double Wi, F00, F01, F10, F11, det, up, plf, smm;
	long i, j, k, np, n[2];
	if (!PyArg_ParseTuple(args, "dOOOO", &Wi, &gradub, &gradlt, &F, &P))
			return NULL;
	np = P->dimensions[0];
	n[0]=2;
	n[1]=2;
	Finv = PyArray_ZEROS(2, &n, NPY_FLOAT64, 0);
	fmm = PyArray_ZEROS(2, &n, NPY_FLOAT64, 0);
	out = PyArray_ZEROS(3, PyArray_DIMS(P), NPY_FLOAT64, 0);
	for (k=0;k<np;k++){
		F00 = *(double *)PyArray_GETPTR3(F, k, 0, 0);
		F10 = *(double *)PyArray_GETPTR3(F, k, 1, 0);
		F01 = *(double *)PyArray_GETPTR3(F, k, 0, 1);
		F11 = *(double *)PyArray_GETPTR3(F, k, 1, 1);
		det = F00*F11 -F01*F10;
		*(double *)PyArray_GETPTR2(Finv, 0, 0) = F11/det;
		*(double *)PyArray_GETPTR2(Finv, 0, 1) = -F01/det;
		*(double *)PyArray_GETPTR2(Finv, 1, 0) = -F10/det;
		*(double *)PyArray_GETPTR2(Finv, 1, 1) = F00/det;
		for (i=0;i<2;i++){
			for (j=0;j<2;j++){
				*(double *)PyArray_GETPTR2(fmm, i, j) = *(double *)PyArray_GETPTR3(gradlt, k, i, 0) * *(double *)PyArray_GETPTR2(Finv, 0, j) + *(double *)PyArray_GETPTR3(gradlt, k, i, 1) * *(double *)PyArray_GETPTR2(Finv, 1, j);
			}
		}
		for (i=0;i<2;i++){
			for (j=0;j<2;j++){
				up = *(double *)PyArray_GETPTR3(gradub, k, i, 0) * *(double *)PyArray_GETPTR3(P, k, 0, j) + *(double *)PyArray_GETPTR3(gradub, k, i, 1) * *(double *)PyArray_GETPTR3(P, k, 1, j);
				smm = *(double *)PyArray_GETPTR2(fmm, i, 0) * *(double *)PyArray_GETPTR3(P, k, 0, j) + *(double *)PyArray_GETPTR2(fmm, i, 1) * *(double *)PyArray_GETPTR3(P, k, 1, j);
				plf = *(double *)PyArray_GETPTR3(P, k, i, j) - *(double *)PyArray_GETPTR2(Finv, j, i);
				*(double *)PyArray_GETPTR3(out,k, i, j) = up + smm - plf/Wi;
			}
		}
						
	}
	
	return out;
}


static PyObject *
cubstokes2D_derivop(PyObject *self, PyObject *args)
{
	PyArrayObject *obspts, *nodes, *f, *F, *out;
	double eps, mu, dx, dy, dx2, dy2, r2, eps2, val11, val10, val01, val00, dh1, dh2, h2, F00, F01, F10, F11, fx, fy, fdotx, Fdx, Fdy, Ffx, Ffy;
	long i, ns, np, k, onp[3];
	if (!PyArg_ParseTuple(args, "ddOOOO", &eps, &mu, &obspts, &nodes, &f, &F))
			return NULL;
	np = F->dimensions[0];
	ns = nodes->dimensions[0];
	onp[0] = np;
	onp[1] = 2;
	onp[2] = 2;
	out = PyArray_ZEROS(3, &onp, NPY_FLOAT64, 0);
	for (k=0;k<np;k++){

		F00 = *(double *)PyArray_GETPTR3(F, k, 0, 0);
		F10 = *(double *)PyArray_GETPTR3(F, k, 1, 0);
		F01 = *(double *)PyArray_GETPTR3(F, k, 0, 1);
		F11 = *(double *)PyArray_GETPTR3(F, k, 1, 1);

		val00 = 0.0;
		val01 = 0.0;
		val10 = 0.0;
		val11 = 0.0;
		for (i=0;i<ns;i++) {
			dx = *(double *)PyArray_GETPTR2(obspts, k, 0) - *(double *)PyArray_GETPTR2(nodes, i, 0);
			dy = *(double *)PyArray_GETPTR2(obspts, k, 1) - *(double *)PyArray_GETPTR2(nodes, i, 1);
			fx = *(double *)PyArray_GETPTR2(f, i, 0);
			fy = *(double *)PyArray_GETPTR2(f, i, 1);
			dx2 = dx*dx;
			dy2 = dy*dy;
			eps2 = eps*eps;
			r2 = dx2 + dy2 + eps2;
			h2 = 1.0/r2;
			dh2 = -2.0/(r2*r2);
			dh1 = -h2 + dh2*eps2;
			fdotx = dx*fx  + dy*fy;
			Fdx = dx*F00  + dy*F10;
			Fdy = dx*F01 + dy*F11;
			Ffx = fx*F00 + fy*F10;
			Ffy = fx*F01 + fy*F11;
			val00 += dh1*fx*Fdx + dh2*dx*fdotx*Fdx + h2*fdotx*F00 + h2*dx*Ffx;
			val01 += dh1*fx*Fdy + dh2*dx*fdotx*Fdy + h2*fdotx*F01 + h2*dx*Ffy;
			val10 += dh1*fy*Fdx + dh2*dy*fdotx*Fdx + h2*fdotx*F10 + h2*dy*Ffx;
			val11 += dh1*fy*Fdy + dh2*dy*fdotx*Fdy + h2*fdotx*F11 + h2*dy*Ffy;

		}
		*(double *)PyArray_GETPTR3(out,k, 0, 0) = val00/(4*M_PI*mu);
		*(double *)PyArray_GETPTR3(out,k, 0, 1) = val01/(4*M_PI*mu);
		*(double *)PyArray_GETPTR3(out,k, 1, 0) = val10/(4*M_PI*mu);
		*(double *)PyArray_GETPTR3(out,k, 1, 1) = val11/(4*M_PI*mu);
	}

	return out;
}

static PyObject *
cubstokes2D_matinv2x2(PyObject *self, PyObject *args)
{
	PyArrayObject *M, *out;
	double det, M00, M01, M10, M11;
	long k,np,nd[3];
	if (!PyArg_ParseTuple(args, "O", &M))
			return NULL;
	np = M->dimensions[0];
	nd[0]=np;
	nd[1]=2;
	nd[2]=2;
	out = PyArray_ZEROS(3, &nd, NPY_FLOAT64, 0);
	for (k=0;k<np;k++){
		M00 = *(double *)PyArray_GETPTR3(M, k, 0, 0);
		M10 = *(double *)PyArray_GETPTR3(M, k, 1, 0);
		M01 = *(double *)PyArray_GETPTR3(M, k, 0, 1);
		M11 = *(double *)PyArray_GETPTR3(M, k, 1, 1);
		det = M00*M11 -M01*M10;
		*(double *)PyArray_GETPTR3(out, k, 0, 0) = M11/det;
		*(double *)PyArray_GETPTR3(out, k, 0, 1) = -M01/det;
		*(double *)PyArray_GETPTR3(out, k, 1, 0) = -M10/det;
		*(double *)PyArray_GETPTR3(out, k, 1, 1) = M00/det;
	}

	return out;
}


static PyMethodDef CubicStokeslet2DMethods[] = {
     {"stressDeriv", cubstokes2D_stressDeriv, METH_VARARGS,
      "Calculates dP/dt from P and other arguments."
      },
     {"matmult", cubstokes2D_matmult, METH_VARARGS,
      "Performs regularized Stokeslet matrix multiplication against forces."
      },
      {"derivop", cubstokes2D_derivop, METH_VARARGS,
       "Matrix action over polymer grid using the derivative of the kernel."
       },
      {"matinv2x2", cubstokes2D_matinv2x2, METH_VARARGS,
       "Inverse of a 2x2 matrix."
       },
{NULL, NULL, 0, NULL}        /* Sentinel */
};
		

PyMODINIT_FUNC
initCubicStokeslet2D(void)
{
    import_array();
    (void) Py_InitModule("CubicStokeslet2D", CubicStokeslet2DMethods);
}

		
