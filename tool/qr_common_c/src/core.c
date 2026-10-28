#include "qr_common.h"
#include <math.h>
#include "model_data.h"
int qc_predict_model(const qc_model *m,const double x[10],qc_pose *p){
    double out[2];int i,j;if(!m||!x||!p)return QC_BAD_ARGUMENT;
    out[0]=m->ymean[0];out[1]=m->ymean[1];
    for(i=0;i<10;i++){
        if(!isfinite(x[i])||!isfinite(m->scale[i])||m->scale[i]<=0)return QC_NUMERIC_ERROR;
        for(j=0;j<2;j++)out[j]+=(x[i]-m->mean[i])/m->scale[i]*m->coef[i][j];
    }
    p->y_mm=out[0]*.05;p->yaw_deg=out[1]*.01;
    return isfinite(p->y_mm)&&isfinite(p->yaw_deg)?QC_OK:QC_NUMERIC_ERROR;
}
int qc_predict_features(const double x[10],int fold,qc_pose *p){
    if(fold < -1 || fold>18)return QC_BAD_ARGUMENT;
    return qc_predict_model(fold<0?&qc_deployment_model:&qc_fold_models[fold],x,p);
}
const char *qc_status_string(int s){
    switch(s){case QC_OK:return "ok";case QC_BAD_ARGUMENT:return "bad argument";
    case QC_IMAGE_ERROR:return "image decode/size error";case QC_QR_ERROR:return "QR fit/alignment error";
    case QC_EDGE_ERROR:return "insufficient valid edge samples";case QC_NUMERIC_ERROR:return "non-finite numerical result";default:return "unknown error";}
}
