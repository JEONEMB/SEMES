#include "main.h"
#include "y_motor.h"
#include "y_motor_port.h"

uint32_t YMotorPort_Lock(void)
{
    uint32_t key = __get_PRIMASK();
    __disable_irq();
    return key;
}
void YMotorPort_Unlock(uint32_t key) { __set_PRIMASK(key); }

void YMotorPort_Enable(uint8_t enabled)
{
    HAL_GPIO_WritePin(Y_EN_N_GPIO_Port, Y_EN_N_Pin,
                     enabled ? GPIO_PIN_RESET : GPIO_PIN_SET);
}
void YMotorPort_Direction(uint8_t high)
{
    HAL_GPIO_WritePin(Y_DIR_GPIO_Port, Y_DIR_Pin,
                     high ? GPIO_PIN_SET : GPIO_PIN_RESET);
}
uint8_t YMotorPort_Limits(void)
{
    uint32_t pins = GPIOA->IDR;
    return (uint8_t)(((pins & Y_HOME_Pin) != 0U) |
                    (((pins & Y_LIMIT_Pin) != 0U) << 1));
}

void YMotorPort_Abort(void)
{
    TIM3->DIER = 0U;
    CLEAR_BIT(TIM3->CR1, TIM_CR1_CEN);
    /* Forced inactive while AF stays connected: PB4 is driven Low, not floating. */
    MODIFY_REG(TIM3->CCMR1, TIM_CCMR1_OC1M, TIM_OCMODE_FORCED_INACTIVE);
    TIM3->CNT = 0U;
    TIM3->SR = 0U;
    HAL_NVIC_ClearPendingIRQ(TIM3_IRQn);
}

int YMotorPort_Init(void)
{
    HAL_NVIC_DisableIRQ(TIM3_IRQn);
    YMotorPort_Enable(0U);
    YMotorPort_Abort();
    /* This project uses APB1=HCLK, TIMPRE=0. Fail closed if CubeMX changes it;
     * guessing the timer clock would silently change pulse widths and speed. */
    uint32_t clock = HAL_RCC_GetPCLK1Freq();
    if ((RCC->CFGR & RCC_CFGR_PPRE1) || (RCC->DCKCFGR & RCC_DCKCFGR_TIMPRE) ||
        clock < 1000000U || clock % 1000000U != 0U || clock / 1000000U > 65536U) {
        return 0;
    }
    TIM3->CR1 = TIM_CR1_OPM; /* no continuous PWM, no ARR preload */
    TIM3->CR2 = 0U;
    TIM3->SMCR = 0U;
    TIM3->PSC = clock / 1000000U - 1U;
    TIM3->CCMR1 = TIM_OCMODE_FORCED_INACTIVE;
    TIM3->CCER = TIM_CCER_CC1E; /* CH1 active High; other outputs disabled */
    TIM3->ARR = 999U;
    TIM3->CCR1 = 995U;
    TIM3->EGR = TIM_EGR_UG;
    TIM3->SR = 0U;
    HAL_NVIC_ClearPendingIRQ(TIM3_IRQn);
    HAL_NVIC_SetPriority(TIM3_IRQn, 0U, 0U);
    HAL_NVIC_EnableIRQ(TIM3_IRQn);
    return 1;
}

void YMotorPort_Arm(uint32_t period_us)
{
    /* PWM2: Low until CCR1, High for 5us, then overflow returns Low.
     * OPM clears CEN in hardware at overflow, even if ISR delivery is delayed.
     * Exactly one rising edge can occur per arm operation. */
    TIM3->DIER = 0U;
    CLEAR_BIT(TIM3->CR1, TIM_CR1_CEN);
    TIM3->CNT = 0U;
    TIM3->ARR = period_us - 1U;
    TIM3->CCR1 = period_us - Y_MOTOR_PULSE_US;
    MODIFY_REG(TIM3->CCMR1, TIM_CCMR1_OC1M, TIM_OCMODE_PWM2);
    TIM3->EGR = TIM_EGR_UG;
    TIM3->SR = 0U;
    HAL_NVIC_ClearPendingIRQ(TIM3_IRQn);
    TIM3->DIER = TIM_DIER_UIE;
    SET_BIT(TIM3->CR1, TIM_CR1_CEN);
}

int YMotorPort_Completed(void)
{
    if (!(TIM3->DIER & TIM_DIER_UIE) || !(TIM3->SR & TIM_SR_UIF)) { return 0; }
    TIM3->SR = 0U;
    TIM3->DIER = 0U;
    return 1;
}
