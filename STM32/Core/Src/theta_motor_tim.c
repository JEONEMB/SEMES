#include "main.h"
#include "theta_motor.h"
#include "theta_motor_port.h"

uint32_t ThetaMotorPort_Lock(void)
{
    uint32_t key = __get_PRIMASK();
    __disable_irq();
    return key;
}
void ThetaMotorPort_Unlock(uint32_t key) { __set_PRIMASK(key); }

void ThetaMotorPort_Enable(uint8_t enabled)
{
#if APP_THETA_ENABLED
    HAL_GPIO_WritePin(THETA_EN_GPIO_Port, THETA_EN_Pin,
                     (enabled == APP_THETA_EN_ACTIVE_HIGH) ? GPIO_PIN_SET : GPIO_PIN_RESET);
#else
    (void)enabled; /* Unknown EN polarity: preserve generated GPIO state. */
#endif
}
void ThetaMotorPort_Direction(uint8_t high)
{
    HAL_GPIO_WritePin(THETA_DIR_GPIO_Port, THETA_DIR_Pin,
                     high ? GPIO_PIN_SET : GPIO_PIN_RESET);
}
uint8_t ThetaMotorPort_RawLimits(void)
{
    return (GPIOA->IDR & THETA_HOME_Pin) != 0U;
}
uint8_t ThetaMotorPort_Limits(void) { return 0U; }

void ThetaMotorPort_Abort(void)
{
    TIM1->DIER = 0U;
    CLEAR_BIT(TIM1->CR1, TIM_CR1_CEN);
    /* Forced inactive while AF stays connected: PA8 is driven Low, not floating. */
    MODIFY_REG(TIM1->CCMR1, TIM_CCMR1_OC1M, TIM_OCMODE_FORCED_INACTIVE);
    TIM1->CNT = 0U;
    TIM1->SR = 0U;
    HAL_NVIC_ClearPendingIRQ(TIM1_UP_TIM10_IRQn);
}

int ThetaMotorPort_Init(void)
{
    HAL_NVIC_DisableIRQ(TIM1_UP_TIM10_IRQn);
    ThetaMotorPort_Enable(0U);
    ThetaMotorPort_Abort();
    if (!APP_THETA_ENABLED) return 0;
    /* This project uses APB2=HCLK, TIMPRE=0. Fail closed if CubeMX changes it;
     * guessing the timer clock would silently change pulse widths and speed. */
    uint32_t clock = HAL_RCC_GetPCLK2Freq();
    if ((RCC->CFGR & RCC_CFGR_PPRE2) || (RCC->DCKCFGR & RCC_DCKCFGR_TIMPRE) ||
        clock < 1000000U || clock % 1000000U != 0U || clock / 1000000U > 65536U) {
        return 0;
    }
    TIM1->CR1 = TIM_CR1_OPM; /* no continuous PWM, no ARR preload */
    TIM1->CR2 = 0U;
    TIM1->SMCR = 0U;
    TIM1->RCR = 0U;
    TIM1->BDTR = TIM_BDTR_MOE;
    TIM1->PSC = clock / 1000000U - 1U;
    TIM1->CCMR1 = TIM_OCMODE_FORCED_INACTIVE;
    TIM1->CCER = TIM_CCER_CC1E; /* CH1 active High; other outputs disabled */
    TIM1->ARR = 999U;
    TIM1->CCR1 = 995U;
    TIM1->EGR = TIM_EGR_UG;
    TIM1->SR = 0U;
    HAL_NVIC_ClearPendingIRQ(TIM1_UP_TIM10_IRQn);
    HAL_NVIC_SetPriority(TIM1_UP_TIM10_IRQn, 0U, 0U);
    HAL_NVIC_EnableIRQ(TIM1_UP_TIM10_IRQn);
    return 1;
}

void ThetaMotorPort_Arm(uint32_t period_us)
{
    /* PWM2: Low until CCR1, High for 5us, then overflow returns Low.
     * OPM clears CEN in hardware at overflow, even if ISR delivery is delayed.
     * Exactly one rising edge can occur per arm operation. */
    TIM1->DIER = 0U;
    CLEAR_BIT(TIM1->CR1, TIM_CR1_CEN);
    TIM1->CNT = 0U;
    TIM1->ARR = period_us - 1U;
    TIM1->CCR1 = period_us - THETA_MOTOR_PULSE_US;
    MODIFY_REG(TIM1->CCMR1, TIM_CCMR1_OC1M, TIM_OCMODE_PWM2);
    TIM1->EGR = TIM_EGR_UG;
    TIM1->SR = 0U;
    HAL_NVIC_ClearPendingIRQ(TIM1_UP_TIM10_IRQn);
    TIM1->DIER = TIM_DIER_UIE;
    SET_BIT(TIM1->CR1, TIM_CR1_CEN);
}

int ThetaMotorPort_Completed(void)
{
    if (!(TIM1->DIER & TIM_DIER_UIE) || !(TIM1->SR & TIM_SR_UIF)) { return 0; }
    TIM1->SR = 0U;
    TIM1->DIER = 0U;
    return 1;
}
