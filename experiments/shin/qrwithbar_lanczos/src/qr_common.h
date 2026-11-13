#ifndef QR_COMMON_H
#define QR_COMMON_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define QC_FEATURES 10
#define QC_POINTS 630
typedef enum { QC_OK=0, QC_BAD_ARGUMENT=1, QC_IMAGE_ERROR=2,
    QC_QR_ERROR=3, QC_EDGE_ERROR=4, QC_NUMERIC_ERROR=5 } qc_status;
typedef struct { const uint8_t *data; int width,height,stride; } qc_gray;
typedef struct { double x,y; int qr; } qc_point;
typedef struct { double x,y,sigma,contrast,rmse; int valid; } qc_edge_sample;
typedef struct { double y_mm,yaw_deg; } qc_pose;
typedef struct {
    /* Ldx,Ldy,Cdx,Cdy,Rdx,Rdy,upper position/slope,lower position/slope.
       Edge slope is height in pixels per normalized x=(x-W/2)/(W/2). */
    double features[QC_FEATURES];
    qc_pose pose;
    int qr_points,upper_valid,lower_valid;
    double upper_line_rms,lower_line_rms;
} qc_result;
typedef struct { double mean[10],scale[10],ymean[2],coef[10][2]; } qc_model;
/* A point-estimation API: no actuator commands, no origin-convergence claim.
   Finite input only. fold=-1 selects deployment fit; 0..18 are evaluation fits. */
int qc_predict_features(const double features[10],int fold,qc_pose *pose);
int qc_predict_model(const qc_model *model,const double features[10],qc_pose *pose);
int qc_extract_edges(const qc_gray *image,const qc_point *points,int count,
    double edge_features[4],qc_edge_sample samples[120],qc_result *diagnostics);
int qc_fit_corner(const qc_gray *image,double sx,double sy,const double bits[4],
    const double basis[4],double *x,double *y,double *rmse);
/* OpenCV adapter only: decodes an image and computes coarse affine QR seeds.
   All subpixel photometric corner/edge fitting and final regression are C. */
int qc_predict_image(const char *utf8_path,const char *calibration_dir,int fold,qc_result *result);
const char *qc_status_string(int status);
#ifdef __cplusplus
}
#endif
#endif
