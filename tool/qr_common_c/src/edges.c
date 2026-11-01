#include "qr_common.h"
#include "numeric.h"
#include "reference_data.h"
#include <stdlib.h>
typedef struct {double x,y,score;} candidate;
typedef struct {candidate c[4];int count;} candidate_group;
typedef struct {const qc_gray *im;int x,cy;} edge_context;
static double edge_eval(const double *p,int i,void *v,double *j){
    edge_context *c=(edge_context*)v;double n=i/7-7,t=i%7-3,z=(n-p[0])/p[1],F=qc_phi(z),g=qc_density(z)/p[1];
    j[0]=-p[2]*g;j[1]=-p[2]*z*g;j[2]=F;j[3]=1;j[4]=n;j[5]=t;
    return p[3]+p[4]*n+p[5]*t+p[2]*F-c->im->data[(c->cy+(int)n)*c->im->stride+c->x+(int)t];
}
static int edge_refine(const qc_gray *im,double x,double y,int polarity,qc_edge_sample *s){
    edge_context ctx;double first=0,last=0,p[6],lo[]={-4,.15,0,-50,-12,-12},hi[]={4,5,0,310,12,12},ss=0;
    int n,t,i,ok;ctx.im=im;ctx.x=qc_round(x);ctx.cy=qc_round(y);
    memset(s,0,sizeof(*s));s->x=ctx.x;
    if(ctx.x<3||ctx.x+3>=im->width||ctx.cy<7||ctx.cy+7>=im->height)return 0;
    for(n=-7;n<=-5;n++)for(t=-3;t<=3;t++)first+=im->data[(ctx.cy+n)*im->stride+ctx.x+t]/21.;
    for(n=5;n<=7;n++)for(t=-3;t<=3;t++)last+=im->data[(ctx.cy+n)*im->stride+ctx.x+t]/21.;
    p[0]=y-ctx.cy;p[1]=1;p[2]=polarity*qc_max(10,polarity*(last-first));p[3]=first;p[4]=p[5]=0;
    lo[2]=polarity>0?1:-300;hi[2]=polarity>0?300:-1;
    ok=qc_optimize(p,6,105,lo,hi,1,edge_eval,&ctx);
    for(i=0;i<105;i++){double jac[6],r=edge_eval(p,i,&ctx,jac);ss+=r*r;}
    s->y=ctx.cy+p[0];s->sigma=p[1];s->contrast=p[2];s->rmse=sqrt(ss/105);
    s->valid=ok&&fabs(p[0])<3.9&&p[1]<4.9&&fabs(p[2])>5&&s->rmse<15;
    return s->valid;
}
/* Direct brightness least-squares hypotheses; no image derivative detector. */
static int edge_candidates(const qc_gray *im,int x,int lo,int hi,int polarity,double prior,candidate_group *group){
    static const double widths[]={.65,1.2,2.2,3.5};
    double scores[2160],values[2160],Fs[4][15],bases[4][9];int y,n,t,k,j,i,length=0;
    if(lo<10)lo=10;if(hi>im->height-10)hi=im->height-10;
    if(x<3||x+3>=im->width||hi<=lo||hi-lo>=2160)return 0;
    for(k=0;k<4;k++){
        memset(bases[k],0,sizeof(bases[k]));
        for(n=-7;n<=7;n++){
            double f=qc_phi(n/widths[k]),a[3]={1,(double)n,f};Fs[k][n+7]=f;
            for(i=0;i<3;i++)for(j=0;j<3;j++)bases[k][i*3+j]+=a[i]*a[j];
        }
    }
    for(y=lo;y<=hi;y+=2){
        double raw[15],sum=0,sum_n=0,null_sse=0,best=-1e300;
        for(n=-7;n<=7;n++){
            raw[n+7]=0;for(t=-3;t<=3;t++)raw[n+7]+=im->data[(y+n)*im->stride+x+t]/7.;
            sum+=raw[n+7];sum_n+=n*raw[n+7];
        }
        for(n=-7;n<=7;n++){double r=raw[n+7]-sum/15-n*sum_n/280;null_sse+=r*r/15;}
        for(k=0;k<4;k++){
            double a[9],rhs[3]={sum,sum_n,0},coef[3]={0},sse=0,score;
            for(n=0;n<15;n++)rhs[2]+=Fs[k][n]*raw[n];memcpy(a,bases[k],sizeof(a));
            if(!qc_solve(3,a,rhs,coef)||polarity*coef[2]<5)continue;
            for(n=-7;n<=7;n++){double r=raw[n+7]-coef[0]-coef[1]*n-coef[2]*Fs[k][n+7];sse+=r*r/15;}
            score=log1p(qc_max(null_sse-sse,0)/(sse+2.25))*sqrt(fabs(coef[2]));
            score-=.5*pow((y-prior)/qc_max(12,.10*(hi-lo)),2);best=qc_max(best,score);
        }
        scores[length]=best;values[length++]=y;
    }
    group->count=0;
    while(group->count<4){
        int best=-1,good=1;double score=-1e299;
        for(i=0;i<length;i++)if(scores[i]>score){score=scores[i];best=i;}
        if(best<0)break;
        for(j=0;j<group->count;j++)if(fabs(values[best]-group->c[j].y)<10)good=0;
        if(good){candidate *c=&group->c[group->count++];c->x=x;c->y=values[best];c->score=qc_max(score,.01);}
        scores[best]=-1e300;
    }
    return group->count>0;
}
static uint32_t next_random(uint32_t *state){uint32_t x=*state;x^=x<<13;x^=x>>17;x^=x<<5;*state=x;return x;}
static int consensus(const candidate_group *g,double width,double *intercept,double *slope){
    uint32_t state=271027;double best=-1;int proposal,i,j,a,b,k,q;
    for(proposal=0;proposal<2800;proposal++){
        const candidate *ca,*cb;double m,h,score=0;
        i=(int)(next_random(&state)%30);j=30+(int)(next_random(&state)%30);
        a=(int)(next_random(&state)%(uint32_t)g[i].count);b=(int)(next_random(&state)%(uint32_t)g[j].count);
        ca=&g[i].c[a];cb=&g[j].c[b];m=(cb->y-ca->y)/(cb->x-ca->x);if(fabs(m)>.06)continue;
        h=ca->y-m*(ca->x-width/2);
        for(k=0;k<60;k++){
            double v=0;for(q=0;q<g[k].count;q++){
                double d=g[k].c[q].y-h-m*(g[k].c[q].x-width/2);
                v=qc_max(v,g[k].c[q].score*exp(-.5*d*d/4));
            }score+=v;
        }
        if(score>best){best=score;*intercept=h;*slope=m;}
    }
    return best>=0;
}
static int robust_line(const qc_edge_sample *s,int width,double *coef,double *rms){
    double weights[60];int i,j,valid=0;
    for(i=0;i<60;i++){weights[i]=s[i].valid/pow(qc_max(s[i].rmse,1.5),2);valid+=s[i].valid;}
    if(valid<12)return 0;
    for(j=0;j<10;j++){
        double a[4]={0},b[2]={0};
        for(i=0;i<60;i++){
            double x=(s[i].x-width/2.)/(width/2.),w=weights[i];
            a[0]+=w;a[1]+=w*x;a[2]+=w*x;a[3]+=w*x*x;b[0]+=w*s[i].y;b[1]+=w*x*s[i].y;
        }
        if(!qc_solve(2,a,b,coef))return 0;
        for(i=0;i<60;i++){
            double r=s[i].y-coef[0]-coef[1]*(s[i].x-width/2.)/(width/2.);
            weights[i]=s[i].valid/pow(qc_max(s[i].rmse,1.5),2)*qc_min(1,.5/qc_max(fabs(r),1e-12));
        }
    }
    *rms=0;for(i=0;i<60;i++)if(s[i].valid){double r=s[i].y-coef[0]-coef[1]*(s[i].x-width/2.)/(width/2.);*rms+=r*r;}
    *rms=sqrt(*rms/valid);return valid;
}
static double median3(double a,double b,double c){return a>b?(b>c?b:qc_min(a,c)):(a>c?a:qc_min(b,c));}
int qc_extract_edges(const qc_gray *im,const qc_point *points,int count,double feat[4],qc_edge_sample samples[120],qc_result *diagnostics){
    double x0[3]={1e9,1e9,1e9},x1[3]={-1e9,-1e9,-1e9},y0[3]={1e9,1e9,1e9},y1[3]={-1e9,-1e9,-1e9};
    double qw,qh,top,bottom,left_end,right_start,qcx,qcy,dx,dy;int i,e;
    if(!im||!im->data||im->width!=3840||im->height!=2160||im->stride<im->width||!points||count!=630||!feat||!samples||!diagnostics)return QC_BAD_ARGUMENT;
    for(i=0;i<count;i++){
        int q=points[i].qr;if(q<0||q>2||!isfinite(points[i].x)||!isfinite(points[i].y))return QC_BAD_ARGUMENT;
        x0[q]=qc_min(x0[q],points[i].x);x1[q]=qc_max(x1[q],points[i].x);y0[q]=qc_min(y0[q],points[i].y);y1[q]=qc_max(y1[q],points[i].y);
    }
    qw=median3(x1[0]-x0[0],x1[1]-x0[1],x1[2]-x0[2]);qh=median3(y1[0]-y0[0],y1[1]-y0[1],y1[2]-y0[2]);
    top=qc_min(y0[0],qc_min(y0[1],y0[2]));bottom=qc_max(y1[0],qc_max(y1[1],y1[2]));
    left_end=qc_min(x0[0],qc_min(x0[1],x0[2]))-1.7*qw;right_start=qc_max(x1[0],qc_max(x1[1],x1[2]))+1.7*qw;
    qcx=(qc_min(x0[0],qc_min(x0[1],x0[2]))+qc_max(x1[0],qc_max(x1[1],x1[2])))/2;qcy=(top+bottom)/2;
    dx=qcx-qc_reference_qr_center[0];dy=qcy-qc_reference_qr_center[1];
    if(qw<10||qh<10||left_end<.065*im->width||right_start>.935*im->width)return QC_EDGE_ERROR;
    for(e=0;e<2;e++){
        candidate_group groups[60];double columns[60],initial=0,slope=0,coef[2],rms;int valid;
        int lo=(int)(e?bottom+.05*qh:qc_max(20,top-2.7*qh));
        int hi=(int)(e?qc_min(im->height-20,bottom+1.25*qh):top-.35*qh),polarity=e?-1:1;
        for(i=0;i<60;i++){
            double x,prior;
            if(i<30)x=.065*im->width+(qc_min(left_end,.45*im->width)-.065*im->width)*i/29.;
            else x=qc_max(right_start,.55*im->width)+(.935*im->width-qc_max(right_start,.55*im->width))*(i-30)/29.;
            columns[i]=x;prior=qc_reference_edges[e*2]+qc_reference_edges[e*2+1]/(im->width/2.)*(x-dx-im->width/2.)+dy;
            if(!edge_candidates(im,qc_round(x),lo,hi,polarity,prior,&groups[i]))return QC_EDGE_ERROR;
        }
        if(!consensus(groups,im->width,&initial,&slope))return QC_EDGE_ERROR;
        for(i=0;i<60;i++)edge_refine(im,columns[i],initial+slope*(columns[i]-im->width/2.),polarity,&samples[e*60+i]);
        valid=robust_line(samples+e*60,im->width,coef,&rms);if(!valid)return QC_EDGE_ERROR;
        feat[e*2]=coef[0]-qc_reference_edges[e*2];feat[e*2+1]=coef[1]-qc_reference_edges[e*2+1];
        if(e==0){diagnostics->upper_valid=valid;diagnostics->upper_line_rms=rms;}
        else{diagnostics->lower_valid=valid;diagnostics->lower_line_rms=rms;}
    }
    return QC_OK;
}
