#ifndef APP_H
#define APP_H

#include "main.h"

void App_Init(UART_HandleTypeDef *uart);
void App_Poll(void);
void App_UartRxComplete(UART_HandleTypeDef *uart);
void App_UartError(UART_HandleTypeDef *uart);

#endif
