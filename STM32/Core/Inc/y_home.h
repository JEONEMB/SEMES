#ifndef Y_HOME_H
#define Y_HOME_H
#include <stdint.h>
typedef enum { Y_HOME_IDLE, Y_HOME_RUNNING, Y_HOME_OK, Y_HOME_STOPPED,
               Y_HOME_FAULT, Y_HOME_LIMIT } YHomeState;
void YHome_Init(void);
int YHome_Start(void);
void YHome_Poll(void);
void YHome_Stop(void);
YHomeState YHome_State(void);
#endif
