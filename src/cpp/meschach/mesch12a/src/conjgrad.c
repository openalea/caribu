
/**************************************************************************
**
** Copyright (C) 1993 David E. Steward & Zbigniew Leyk, all rights reserved.
**
**			     Meschach Library
** 
** This Meschach Library is provided "as is" without any express 
** or implied warranty of any kind with respect to this software. 
** In particular the authors shall not be liable for any direct, 
** indirect, special, incidental or consequential damages arising 
** in any way from use of the software.
** 
** Everyone is granted permission to copy, modify and redistribute this
** Meschach Library, provided:
**  1.  All copies contain this copyright notice.
**  2.  All modified copies shall carry a notice stating who
**      made the last modification and the date of such modification.
**  3.  No charge is made for this software or works derived from it.  
**      This clause shall not be construed as constraining other software
**      distributed on the same medium as this software, nor is a
**      distribution fee considered a charge.
**
***************************************************************************/


/*
	Conjugate gradient routines file
	Uses sparse matrix input & sparse Cholesky factorisation in pccg().

	All the following routines use routines to define a matrix
		rather than use any explicit representation
		(with the exeception of the pccg() pre-conditioner)
	The matrix A is defined by

		VEC *(*A)(void *params, VEC *x, VEC *y)

	where y = A.x on exit, and y is returned. The params argument is
	intended to make it easier to re-use & modify such routines.

	If we have a sparse matrix data structure
		SPMAT	*A_mat;
	then these can be used by passing sp_mv_mlt as the function, and
	A_mat as the param.
*/

/* conjgrad.c — C23-clean version */

#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "sparse.h"

static char rcsid[] = "$Id: conjgrad.c,v 1.4 1994/01/13 05:36:45 des Exp $";

/* Max iterations control */
static int max_iter = 10000;
int cg_num_iters;

/* Matrix-as-routine type definition: A(params, in, out) -> out */
typedef VEC *(*MTX_FN)(void *params, VEC *in, VEC *out);

/* Prototypes for adapters from Meschach's SPMAT* signatures to MTX_FN */
static VEC *sp_mv_mlt_adapter(void *A, VEC *x, VEC *out);
static VEC *sp_vm_mlt_adapter(void *A, VEC *x, VEC *out);
static VEC *spCHsolve_adapter(void *LLT, VEC *x, VEC *out);

/* External functions are declared in sparse.h / matrix.h:
      VEC *sp_mv_mlt(SPMAT *A, VEC *x, VEC *out);
      VEC *sp_vm_mlt(SPMAT *A, VEC *x, VEC *out);
      VEC *spCHsolve(SPMAT *LLT, VEC *b, VEC *x);
   We only provide the adapters with the (void*, VEC*, VEC*) ABI expected by MTX_FN.
*/

/* Set/get maximum iterations: if numiter >= 2 set it; return previous value */
int cg_set_maxiter(int numiter)
{
    int temp;

    if (numiter < 2)
        return max_iter;
    temp = max_iter;
    max_iter = numiter;
    return temp;
}

/* -------------------- Preconditioned Conjugate Gradient -------------------- */
/* pccg -- solves A.x = b using pre-conditioner M_inv
   results are stored in x (if x != NULL), which is returned */
VEC *pccg(MTX_FN A, void *A_params,
          MTX_FN M_inv, void *M_params,
          VEC *b, double eps, VEC *x)
{
    VEC *r = VNULL, *p = VNULL, *q = VNULL, *z = VNULL;
    int k;
    Real alpha, beta, ip, old_ip, norm_b;

    if (!A || !b)
        error(E_NULL, "pccg");
    if (x == b)
        error(E_INSITU, "pccg");

    x = v_resize(x, b->dim);
    if (eps <= 0.0)
        eps = MACHEPS;

    r = v_get(b->dim);
    p = v_get(b->dim);
    q = v_get(b->dim);
    z = v_get(b->dim);

    norm_b = v_norm2(b);

    v_zero(x);
    r = v_copy(b, r);
    old_ip = 0.0;

    for (k = 0;; k++)
    {
        if (v_norm2(r) < eps * norm_b)
            break;
        if (k > max_iter)
            error(E_ITER, "pccg");

        if (M_inv)
            M_inv(M_params, r, z);
        else
            v_copy(r, z); /* M == identity */

        ip = in_prod(z, r);

        if (k)
        {
            beta = ip / old_ip;
            p = v_mltadd(z, p, beta, p);
        }
        else
        {
            beta = 0.0;
            p = v_copy(z, p);
            old_ip = 0.0;
        }

        q = A(A_params, p, q);
        alpha = ip / in_prod(p, q);
        x = v_mltadd(x, p, alpha, x);
        r = v_mltadd(r, q, -alpha, r);
        old_ip = ip;
    }

    cg_num_iters = k;

    V_FREE(p);
    V_FREE(q);
    V_FREE(r);
    V_FREE(z);

    return x;
}

/* sp_pccg -- convenience wrapper for SPMAT data structures */
VEC *sp_pccg(SPMAT *A, SPMAT *LLT, VEC *b, double eps, VEC *x)
{
    return pccg(sp_mv_mlt_adapter, (void *)A,
                spCHsolve_adapter, (void *)LLT,
                b, eps, x);
}

/* -------------------- CGS (Conjugate Gradient Squared) -------------------- */
/* cgs -- computes a solution x to A.x=b using the CGS algorithm */
VEC *cgs(MTX_FN A, void *A_params, VEC *b, VEC *r0, double tol, VEC *x)
{
    VEC *p, *q, *r, *u, *v, *tmp1, *tmp2;
    Real alpha, beta, norm_b, rho, old_rho, sigma;
    int iter;

    if (!A || !x || !b || !r0)
        error(E_NULL, "cgs");
    if (x->dim != b->dim || r0->dim != x->dim)
        error(E_SIZES, "cgs");
    if (tol <= 0.0)
        tol = MACHEPS;

    p = v_get(x->dim);
    q = v_get(x->dim);
    r = v_get(x->dim);
    u = v_get(x->dim);
    v = v_get(x->dim);
    tmp1 = v_get(x->dim);
    tmp2 = v_get(x->dim);

    norm_b = v_norm2(b);
    A(A_params, x, tmp1);
    v_sub(b, tmp1, r);
    v_zero(p);
    v_zero(q);
    old_rho = 1.0;

    iter = 0;
    while (v_norm2(r) > tol * norm_b)
    {
        if (++iter > max_iter)
            break; /* was error(E_ITER,"cgs"); */

        rho = in_prod(r0, r);
        if (old_rho == 0.0)
            error(E_SING, "cgs");
        beta = rho / old_rho;

        v_mltadd(r, q, beta, u);
        v_mltadd(q, p, beta, tmp1);
        v_mltadd(u, tmp1, beta, p);

        A(A_params, p, v);

        sigma = in_prod(r0, v);
        if (sigma == 0.0)
            error(E_SING, "cgs");
        alpha = rho / sigma;

        v_mltadd(u, v, -alpha, q);
        v_add(u, q, tmp1);

        A(A_params, tmp1, tmp2);

        v_mltadd(r, tmp2, -alpha, r);
        v_mltadd(x, tmp1, alpha, x);

        old_rho = rho;
    }
    cg_num_iters = iter;

    V_FREE(p); V_FREE(q); V_FREE(r);
    V_FREE(u); V_FREE(v);
    V_FREE(tmp1); V_FREE(tmp2);

    return x;
}

/* sp_cgs -- convenience wrapper for SPMAT data structures */
VEC *sp_cgs(SPMAT *A, VEC *b, VEC *r0, double tol, VEC *x)
{
    return cgs(sp_mv_mlt_adapter, (void *)A, b, r0, tol, x);
}

/* -------------------- LSQR (Paige & Saunders) -------------------- */
/* lsqr -- finds min_x ||A.x - b||_2 using A and AT function operators */
VEC *lsqr(MTX_FN A, MTX_FN AT, void *A_params, VEC *b, double tol, VEC *x)
{
    VEC *u, *v, *w, *tmp;
    Real alpha, beta, norm_b, phi, phi_bar,
         rho, rho_bar, rho_max, theta;
    Real s, c; /* Givens rotations */
    int iter, m, n;

    if (!b || !x)
        error(E_NULL, "lsqr");
    if (tol <= 0.0)
        tol = MACHEPS;

    m = b->dim; n = x->dim;
    u = v_get((u_int)m);
    v = v_get((u_int)n);
    w = v_get((u_int)n);
    tmp = v_get((u_int)n);
    norm_b = v_norm2(b);

    v_zero(x);
    beta = v_norm2(b);
    if (beta == 0.0)
    {
        V_FREE(tmp); V_FREE(u); V_FREE(v); V_FREE(w);
        return x;
    }

    sv_mlt(1.0 / beta, b, u);
    tracecatch(AT(A_params, u, v), "lsqr");
    alpha = v_norm2(v);
    if (alpha == 0.0)
    {
        V_FREE(tmp); V_FREE(u); V_FREE(v); V_FREE(w);
        return x;
    }
    sv_mlt(1.0 / alpha, v, v);
    v_copy(v, w);
    phi_bar = beta; rho_bar = alpha;

    rho_max = 1.0;
    iter = 0;
    do {
        if (++iter > max_iter)
            error(E_ITER, "lsqr");

        tmp = v_resize(tmp, m);
        tracecatch(A(A_params, v, tmp), "lsqr");

        v_mltadd(tmp, u, -alpha, u);
        beta = v_norm2(u); sv_mlt(1.0 / beta, u, u);

        tmp = v_resize(tmp, n);
        tracecatch(AT(A_params, u, tmp), "lsqr");
        v_mltadd(tmp, v, -beta, v);
        alpha = v_norm2(v); sv_mlt(1.0 / alpha, v, v);

        rho = sqrt(rho_bar * rho_bar + beta * beta);
        if (rho > rho_max)
            rho_max = rho;
        c = rho_bar / rho;
        s = beta / rho;
        theta   =  s * alpha;
        rho_bar = -c * alpha;
        phi     =  c * phi_bar;
        phi_bar =  s * phi_bar;

        if (rho == 0.0)
            error(E_SING, "lsqr");
        v_mltadd(x, w, phi / rho, x);
        v_mltadd(v, w, -theta / rho, w);
    } while (fabs(phi_bar * alpha * c) > tol * norm_b / rho_max);

    cg_num_iters = iter;

    V_FREE(tmp); V_FREE(u); V_FREE(v); V_FREE(w);
    return x;
}

/* sp_lsqr -- convenience wrapper for SPMAT data structures */
VEC *sp_lsqr(SPMAT *A, VEC *b, double tol, VEC *x)
{
    return lsqr(sp_mv_mlt_adapter, sp_vm_mlt_adapter, (void *)A, b, tol, x);
}

/* -------------------- Adapters -------------------- */

static VEC *sp_mv_mlt_adapter(void *A, VEC *x, VEC *out)
{
    return sp_mv_mlt((SPMAT *)A, x, out);
}

static VEC *sp_vm_mlt_adapter(void *A, VEC *x, VEC *out)
{
    return sp_vm_mlt((SPMAT *)A, x, out);
}

static VEC *spCHsolve_adapter(void *LLT, VEC *b, VEC *x)
{
    return spCHsolve((SPMAT *)LLT, b, x);
}

