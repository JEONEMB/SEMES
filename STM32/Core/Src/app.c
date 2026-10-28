#include "app.h"
#include "app_config.h"
#include "robot_stage_service.h"

static UART_HandleTypeDef *link;
static uint8_t rx_byte;
/* Arrival timestamps preserve inter-byte timeout while TX/backlog waits. */
static uint8_t rx_queue[APP_RX_CAPACITY];
static uint32_t rx_time[APP_RX_CAPACITY];
static volatile uint16_t rx_head, rx_tail;
static volatile uint8_t rx_fault, rx_restart;
static volatile uint32_t uart_errors, rx_overflows;
static uint32_t tx_errors;
static RsParser parser;
static RsService service;
typedef struct { uint8_t bytes[RS_MAX_FRAME]; uint16_t length; } TxPacket;
static TxPacket tx_queue[APP_TX_QUEUE_CAPACITY];
static unsigned tx_head, tx_tail, tx_count;
static uint8_t tx_active;
static uint32_t last_sample;
static uint8_t switches_raw, switches_filtered, switches_valid;
static uint8_t candidate[3], sample_count[3];
static uint8_t limit_known, limit_previous, limit_pending;

static uint32_t Lock(void)
{
    uint32_t state = __get_PRIMASK();
    __disable_irq();
    return state;
}
static void Unlock(uint32_t state) { __set_PRIMASK(state); }

static void DisablePulses(void)
{
    HAL_GPIO_WritePin(Y_EN_N_GPIO_Port, Y_EN_N_Pin, GPIO_PIN_SET);
    CLEAR_BIT(TIM1->CCER, TIM_CCER_CC1E | TIM_CCER_CC1NE);
    CLEAR_BIT(TIM1->BDTR, TIM_BDTR_MOE);
    CLEAR_BIT(TIM1->CR1, TIM_CR1_CEN);
    CLEAR_BIT(TIM3->CCER, TIM_CCER_CC1E);
    CLEAR_BIT(TIM3->CR1, TIM_CR1_CEN);
}

static void PollSwitches(uint32_t now)
{
    uint32_t elapsed = now - last_sample;
    if (elapsed == 0U) { return; }
    last_sample = now;
    uint32_t pins = Y_HOME_GPIO_Port->IDR;
    switches_raw = (uint8_t)(((pins & Y_HOME_Pin) != 0U) |
                   (((pins & Y_LIMIT_Pin) != 0U) << 1) |
                   (((pins & THETA_HOME_Pin) != 0U) << 2));
    for (unsigned i = 0; i < 3U; ++i) {
        uint8_t bit = (uint8_t)(1U << i);
        uint8_t value = (uint8_t)((switches_raw >> i) & 1U);
        if (elapsed != 1U || candidate[i] != value) {
            candidate[i] = value;
            sample_count[i] = 0U;
            switches_valid &= (uint8_t)~bit;
        }
        if (sample_count[i] < APP_SWITCH_STABLE_SAMPLES) { ++sample_count[i]; }
        if (sample_count[i] == APP_SWITCH_STABLE_SAMPLES) {
            switches_filtered = (uint8_t)((switches_filtered & ~bit) | (value << i));
            switches_valid |= bit;
            /* Initial levels establish a baseline; only new Y limit edges emit.
             * Theta HOME is not an end-of-travel limit. */
            if (i < 2U) {
                if ((limit_known & bit) && !(limit_previous & bit) && value) {
                    limit_pending = 1U;
                }
                limit_known |= bit;
                limit_previous = (uint8_t)((limit_previous & ~bit) | (value << i));
            }
        }
    }
}

static void QueueReply(const RsFrame *reply)
{
    /* Main context only; caller reserves a free slot before dispatch. */
    tx_queue[tx_head].length = (uint16_t)RsFrame_Encode(reply, tx_queue[tx_head].bytes);
    tx_head = (tx_head + 1U) % APP_TX_QUEUE_CAPACITY;
    ++tx_count;
}

static void PollTx(void)
{
    if (link->gState != HAL_UART_STATE_READY) { return; }
    if (tx_active) {
        tx_active = 0U;
        tx_tail = (tx_tail + 1U) % APP_TX_QUEUE_CAPACITY;
        --tx_count;
    }
    if (tx_count) {
        HAL_StatusTypeDef result = HAL_UART_Transmit_IT(link,
            tx_queue[tx_tail].bytes, tx_queue[tx_tail].length);
        if (result == HAL_OK) { tx_active = 1U; }
        else if (result != HAL_BUSY) { ++tx_errors; }
    }
}

void App_Init(UART_HandleTypeDef *uart)
{
    link = uart;
    rx_head = rx_tail = 0U;
    rx_fault = rx_restart = 0U;
    uart_errors = rx_overflows = tx_errors = 0U;
    tx_head = tx_tail = tx_count = 0U;
    tx_active = 0U;
    switches_raw = switches_filtered = switches_valid = 0U;
    limit_known = limit_previous = limit_pending = 0U;
    for (unsigned i = 0; i < 3U; ++i) { candidate[i] = sample_count[i] = 0U; }
    RsParser_Reset(&parser);
    RsService_Init(&service);
    /* Keep PC7 at generated Low; its physical disable meaning is UNKNOWN.
     * Motor 24V must remain off during bring-up. */
    DisablePulses();
    last_sample = HAL_GetTick();
    RsFrame boot;
    RsService_Event(&service, RS_BOOT, &boot);
    QueueReply(&boot);
    if (HAL_UART_Receive_IT(link, &rx_byte, 1U) != HAL_OK) {
        rx_restart = 1U;
        ++uart_errors;
    }
}

void App_UartRxComplete(UART_HandleTypeDef *uart)
{
    if (uart != link) { return; }
    uint16_t next = (uint16_t)((rx_head + 1U) % APP_RX_CAPACITY);
    /* HAL can deliver RX before its error callback; rearming here would
     * clear ErrorCode before HAL has handled the fault. */
    if (uart->ErrorCode != HAL_UART_ERROR_NONE) {
        rx_fault = rx_restart = 1U;
        return;
    }
    if (!rx_fault) {
        if (next == rx_tail) { rx_fault = 1U; ++rx_overflows; }
        else {
            rx_queue[rx_head] = rx_byte;
            rx_time[rx_head] = HAL_GetTick();
            rx_head = next;
        }
    }
    if (HAL_UART_Receive_IT(link, &rx_byte, 1U) != HAL_OK) {
        rx_restart = rx_fault = 1U;
    }
}

void App_UartError(UART_HandleTypeDef *uart)
{
    if (uart != link) { return; }
    ++uart_errors;
    rx_fault = rx_restart = 1U;
}

static void RecoverRx(void)
{
    uint32_t key = Lock();
    if (rx_fault) {
        rx_tail = rx_head;
        RsParser_Reset(&parser);
        rx_fault = 0U;
    }
    if (rx_restart) {
        (void)HAL_UART_AbortReceive(link);
        __HAL_UART_CLEAR_OREFLAG(link);
        if (HAL_UART_Receive_IT(link, &rx_byte, 1U) == HAL_OK) { rx_restart = 0U; }
    }
    Unlock(key);
}

void App_Poll(void)
{
    uint32_t now = HAL_GetTick();
    PollSwitches(now);
    RecoverRx();
    PollTx();
    /* Bounded processing and separate buffers keep RX running during TX.
     * Unlimited host flooding is not supported; overflow resynchronizes. */
    for (unsigned budget = 0; budget < APP_RX_POLL_BUDGET; ++budget) {
        if (tx_count == APP_TX_QUEUE_CAPACITY) { break; }
        RsFrame request, reply;
        if (RsParser_Next(&parser, &request)) {
            if (rx_fault) { RecoverRx(); continue; }
            if (request.type == RS_STOP && request.len == 0U) { DisablePulses(); }
            /* V1 has no filter-valid bits: use raw inputs until stable. */
            uint8_t inputs = (uint8_t)((switches_filtered & switches_valid) |
                                      (switches_raw & ~switches_valid));
            RsService_Handle(&service, &request, inputs, &reply);
            QueueReply(&reply);
            continue;
        }
        uint32_t key = Lock();
        if (rx_fault) { Unlock(key); RecoverRx(); continue; }
        if (rx_tail == rx_head) {
            RsParser_Expire(&parser, HAL_GetTick());
            Unlock(key);
            break;
        }
        uint8_t byte = rx_queue[rx_tail];
        uint32_t arrival = rx_time[rx_tail];
        rx_tail = (uint16_t)((rx_tail + 1U) % APP_RX_CAPACITY);
        Unlock(key);
        RsParser_Push(&parser, byte, arrival);
    }
    if (limit_pending && tx_count < APP_TX_QUEUE_CAPACITY) {
        RsFrame event;
        RsService_Event(&service, RS_LIMIT_HIT, &event);
        QueueReply(&event);
        limit_pending = 0U;
    }
    PollTx();
}
