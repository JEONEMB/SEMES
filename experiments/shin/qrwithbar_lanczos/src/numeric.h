#ifndef QC_NUMERIC_H
#define QC_NUMERIC_H
#include <cmath>
#include <string.h>
using std::isfinite;
#define QC_PI 3.14159265358979323846
static double qc_min(double a,double b){return a<b?a:b;}
static double qc_max(double a,double b){return a>b?a:b;}
static double qc_clip(double x,double a,double b){return qc_min(b,qc_max(a,x));}
/* Python/NumPy use nearest-even rounding for patch centers. */
static int qc_round(double x){double f=floor(x),d=x-f;return (int)(d<.5?f:d>.5?f+1:(fmod(f,2)==0?f:f+1));}
static double qc_phi(double z){return .5*(1+erf(z/sqrt(2.)));}
static double qc_density(double z){return exp(-.5*z*z)/sqrt(2*QC_PI);}
static int qc_solve(int n,double *a,double *b,double *out){
    int i,j,k,p;double t,m;
    for(i=0;i<n;i++){
        p=i;for(j=i+1;j<n;j++)if(fabs(a[j*n+i])>fabs(a[p*n+i]))p=j;
        if(fabs(a[p*n+i])<1e-20)return 0;
        for(k=i;k<n;k++){t=a[i*n+k];a[i*n+k]=a[p*n+k];a[p*n+k]=t;}
        t=b[i];b[i]=b[p];b[p]=t;
        for(j=i+1;j<n;j++){m=a[j*n+i]/a[i*n+i];for(k=i;k<n;k++)a[j*n+k]-=m*a[i*n+k];b[j]-=m*b[i];}
    }
    for(i=n-1;i>=0;i--){t=b[i];for(j=i+1;j<n;j++)t-=a[i*n+j]*out[j];out[i]=t/a[i*n+i];}
    return 1;
}
typedef double (*qc_eval)(const double *,int,void *,double *);
/* Bound-constrained scaled damped Gauss-Newton with analytic Jacobian and
   monotone objective acceptance. robust=1 is soft-L1 with f_scale=3. */
static int qc_optimize(double *p,int n,int m,const double *lo,const double *hi,int robust,qc_eval evaluate,void *ctx){
    double lambda=1e-3,old=1e300;int iter,i,j,k;
    for(iter=0;iter<140;iter++){
        double a[36]={0},g[6]={0},d[6]={0},trial[6],scale[6],cost=0,ac[36],rhs[6],newcost=0,maxstep=0;
        for(i=0;i<m;i++){
            double jac[6],r=evaluate(p,i,ctx,jac),w=robust?1/sqrt(1+r*r/9):1;
            cost+=robust?18*(sqrt(1+r*r/9)-1):r*r;
            for(j=0;j<n;j++){g[j]+=w*jac[j]*r;for(k=0;k<n;k++)a[j*n+k]+=w*jac[j]*jac[k];}
        }
        if(!isfinite(cost))return 0;
        for(j=0;j<n;j++)scale[j]=sqrt(qc_max(a[j*n+j],1e-12));
        for(j=0;j<n;j++){rhs[j]=-g[j]/scale[j];for(k=0;k<n;k++)ac[j*n+k]=a[j*n+k]/scale[j]/scale[k];ac[j*n+j]+=lambda;}
        if(!qc_solve(n,ac,rhs,d))return 0;
        for(j=0;j<n;j++){trial[j]=qc_clip(p[j]+d[j]/scale[j],lo[j],hi[j]);maxstep=qc_max(maxstep,fabs(trial[j]-p[j]));}
        for(i=0;i<m;i++){double jac[6],r=evaluate(trial,i,ctx,jac);newcost+=robust?18*(sqrt(1+r*r/9)-1):r*r;}
        if(newcost<cost){
            memcpy(p,trial,(size_t)n*sizeof(double));lambda=qc_max(lambda*.3,1e-12);
            if(maxstep<1e-8 || fabs(cost-newcost)<1e-11*(1+cost))return 1;
            old=cost;
        }else{lambda*=8;if(lambda>1e14)return isfinite(old)||cost<1e-12;}
    }
    return isfinite(old);
}
#endif
