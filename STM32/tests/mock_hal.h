#ifndef MOCK_HAL_H
#define MOCK_HAL_H
/* Replace app.h's hardware include when compiling app.c for emulation. */
#define APP_H
#define __MAIN_H
#include <stdint.h>
typedef struct { uint32_t ErrorCode, gState; } UART_HandleTypeDef;
typedef struct { volatile uint32_t CCER, BDTR, CR1, CR2, SMCR, DIER,
    CCMR1, CNT, SR, PSC, ARR, CCR1, EGR, RCR; } TestTimer;
typedef struct { uint32_t CFGR, DCKCFGR; } TestRcc;
extern TestRcc test_rcc;
#define RCC (&test_rcc)
#define RCC_CFGR_PPRE1 (7U << 10)
#define RCC_DCKCFGR_TIMPRE (1U << 24)
#define TIM_CR1_OPM 8U
#define TIM_CCMR1_OC1M (7U << 4)
#define TIM_OCMODE_FORCED_INACTIVE (4U << 4)
#define TIM_OCMODE_PWM2 (7U << 4)
#define TIM_EGR_UG 1U
#define TIM_DIER_UIE 1U
#define TIM_SR_UIF 1U
#define TIM3_IRQn 29
#define TIM1_UP_TIM10_IRQn 25
#define RCC_CFGR_PPRE2 (7U << 13)
#define THETA_EN_GPIO_Port (&gpio)
#define THETA_EN_Pin 128U
#define THETA_DIR_GPIO_Port (&gpio)
#define THETA_DIR_Pin 512U
uint32_t HAL_RCC_GetPCLK2Freq(void);
#define SET_BIT(reg, mask) ((reg) |= (mask))
#define MODIFY_REG(reg, mask, value) ((reg) = ((reg) & ~(mask)) | (value))
#define GPIOA (&gpio)
#define Y_DIR_GPIO_Port (&gpio)
#define Y_DIR_Pin 32U
#define GPIO_PIN_RESET 0U
uint32_t HAL_RCC_GetPCLK1Freq(void);
void HAL_NVIC_DisableIRQ(int irq);
void HAL_NVIC_ClearPendingIRQ(int irq);
void HAL_NVIC_SetPriority(int irq, unsigned preemption, unsigned sub);
void HAL_NVIC_EnableIRQ(int irq);
typedef struct { uint32_t IDR; } TestGpio;
typedef enum { HAL_OK, HAL_ERROR, HAL_BUSY } HAL_StatusTypeDef;
extern TestTimer timer1, timer3;
extern TestGpio gpio;
#define TIM1 (&timer1)
#define TIM3 (&timer3)
#define TIM_CCER_CC1E 1U
#define TIM_CCER_CC1NE 4U
#define TIM_BDTR_MOE 0x8000U
#define TIM_CR1_CEN 1U
#define CLEAR_BIT(reg, mask) ((reg) &= ~(mask))
#define Y_HOME_GPIO_Port (&gpio)
#define Y_EN_N_GPIO_Port (&gpio)
#define Y_HOME_Pin 1U
#define Y_LIMIT_Pin 2U
#define THETA_HOME_Pin 16U
#define Y_EN_N_Pin 1024U
#define GPIO_PIN_SET 1U
#define LED2 0U
#define HAL_UART_ERROR_NONE 0U
#define HAL_UART_STATE_READY 0U
#define __HAL_UART_CLEAR_OREFLAG(uart) ((uart)->ErrorCode = 0U)
uint32_t __get_PRIMASK(void);
void __disable_irq(void);
void __set_PRIMASK(uint32_t state);
uint32_t HAL_GetTick(void);
void HAL_GPIO_WritePin(TestGpio *port, uint32_t pin, uint32_t level);
void BSP_LED_Toggle(unsigned led);
HAL_StatusTypeDef HAL_UART_Receive_IT(UART_HandleTypeDef *uart, uint8_t *byte, uint16_t length);
HAL_StatusTypeDef HAL_UART_Transmit_IT(UART_HandleTypeDef *uart, uint8_t *bytes, uint16_t length);
HAL_StatusTypeDef HAL_UART_AbortReceive(UART_HandleTypeDef *uart);
void App_Init(UART_HandleTypeDef *uart);
void App_Poll(void);
void App_UartRxComplete(UART_HandleTypeDef *uart);
void App_UartError(UART_HandleTypeDef *uart);
#endif
