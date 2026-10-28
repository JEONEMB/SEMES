#include "qr_common.h"
#include "numeric.h"
typedef struct {const qc_gray *im;double sx,sy,bits[4],inv[4];int ix,iy;} corner_context;
static double corner_eval(const double *p,int i,void *v,double *j){
    corner_context *c=(corner_context*)v;int x=i%9-4,y=i/9-4;
    double xx=x+c->ix-c->sx-p[0],yy=y+c->iy-c->sy-p[1];
    double u=c->inv[0]*xx+c->inv[1]*yy,vv=c->inv[2]*xx+c->inv[3]*yy;
    double fu=qc_phi(u/p[2]),fv=qc_phi(vv/p[2]),gu=qc_density(u/p[2])/p[2],gv=qc_density(vv/p[2])/p[2];
    double a=c->bits[0],b=c->bits[1],cc=c->bits[2],d=c->bits[3],cross=a-b-cc+d;
    double pattern=a+(b-a)*fu+(cc-a)*fv+cross*fu*fv,du=b-a+cross*fv,dv=cc-a+cross*fu;
    j[0]=p[3]*(du*gu*c->inv[0]+dv*gv*c->inv[2]);
    j[1]=p[3]*(du*gu*c->inv[1]+dv*gv*c->inv[3]);
    j[2]=p[3]*(du*gu*u+dv*gv*vv)/p[2];j[3]=-pattern;j[4]=1;
    return p[4]-p[3]*pattern-c->im->data[(c->iy+y)*c->im->stride+c->ix+x];
}
int qc_fit_corner(const qc_gray *im,double sx,double sy,const double bits[4],const double basis[4],double *ox,double *oy,double *rmse){
    corner_context c;double det,p[5],lo[]={-3.6,-3.6,.18,1,0},hi[]={3.6,3.6,2.6,300,300};
    double minv=255,maxv=0,ss=0;int i,j;
    if(!im||!im->data||im->stride<im->width||!bits||!basis||!ox||!oy||!rmse||!isfinite(sx)||!isfinite(sy))return QC_BAD_ARGUMENT;
    c.im=im;c.sx=sx;c.sy=sy;c.ix=qc_round(sx);c.iy=qc_round(sy);
    if(c.ix<4||c.iy<4||c.ix+4>=im->width||c.iy+4>=im->height)return QC_QR_ERROR;
    det=basis[0]*basis[3]-basis[1]*basis[2];if(fabs(det)<1e-8)return QC_QR_ERROR;
    c.inv[0]=basis[3]/det;c.inv[1]=-basis[1]/det;c.inv[2]=-basis[2]/det;c.inv[3]=basis[0]/det;memcpy(c.bits,bits,4*sizeof(double));
    for(i=-4;i<=4;i++)for(j=-4;j<=4;j++){double z=im->data[(c.iy+i)*im->stride+c.ix+j];minv=qc_min(minv,z);maxv=qc_max(maxv,z);}
    p[0]=p[1]=0;p[2]=.8;p[3]=qc_max(45,maxv-minv);p[4]=maxv;
    if(!qc_optimize(p,5,81,lo,hi,0,corner_eval,&c))return QC_QR_ERROR;
    for(i=0;i<81;i++){double jac[6],r=corner_eval(p,i,&c,jac);ss+=r*r;}
    *ox=sx+p[0];*oy=sy+p[1];*rmse=sqrt(ss/81);
    return QC_OK;
}
