/**
 * @file    node.c
 * @brief   Sensor node in an RPL-based WSN to collect execution and network metrics.
 * 
 * This node listens for probe packets from the root node (PING),
 * measures energy consumption, latency, RSSI, LQI, and network traffic,
 * and responds with a structured metrics packet for multi-objective analysis.
 * 
 * Metrics include energy breakdown (CPU, LPM, TX, RX), packet counters,
 * byte counters, latency estimates, and link quality information.
 */

#include "contiki.h"
#include "net/routing/routing.h"
#include "net/netstack.h"
#include "net/ipv6/simple-udp.h"
#include "net/ipv6/uip-ds6.h"
#include "net/ipv6/uiplib.h"
#include "sys/energest.h"
#include "random.h"
#include <stdio.h>
#include <string.h>           /* memcmp / memcpy */

#include "metrics-packet.h"

/* ---------- Conversão de tempo ------------------------------------------ */
#define NOW_TICKS()         clock_time()                               /* ticks */
#define TICKS_TO_MS(t)      (((uint64_t)(t) * 1000) / CLOCK_SECOND)    /* ms   */

//------------------------ Configuration Constants ----------------------------

#define UDP_CLIENT_PORT   8765
#define UDP_SERVER_PORT   5678

#define BASE_INTERVAL (10 * CLOCK_SECOND)    ///< Base interval between transmissions
#define JITTER        (random_rand() % (CLOCK_SECOND)) ///< Randomized jitter to avoid collisions

// Average power consumption estimates (in mW)
#define CPU_POWER_ACTIVE  1.8
#define LPM_POWER         0.0545
#define RADIO_TX_POWER    17.4
#define RADIO_RX_POWER    19.7

//------------------------ Debug Flags (optional) -----------------------------

//#define DEBUG_ENERGY_TIME_IN_SECONDS
//#define DEBUG_PRINT_METRICS_PACKET
//#define DEBUG_RX_CALLBACK

//------------------------ Global Variables -----------------------------------

static struct simple_udp_connection udp_conn;

static uint64_t root_to_node_latency = 0;
static uint32_t total_sent = 0, total_received = 0;
static uint16_t bytes_tx = 0, bytes_rx = 0;

static radio_value_t last_rssi = 0;
static radio_value_t last_lqi = 0;

//------------------------ Utility Functions ----------------------------------

/**
 * @brief Converts ticks to seconds based on ENERGEST_SECOND.
 */
static float to_seconds(uint64_t time) {
    return (float)time / ENERGEST_SECOND;
}

/**
 * @brief Prints the node's own link-local IPv6 address.
 */
static void print_own_link_local(void) {
    uip_ds6_addr_t *ll_addr = uip_ds6_get_link_local(ADDR_PREFERRED);
    if (ll_addr != NULL) {
        char addr_str[UIPLIB_IPV6_MAX_STR_LEN];
        uiplib_ipaddr_snprint(addr_str, sizeof(addr_str), &ll_addr->ipaddr);
        printf("Sensor IPv6 = %s\n", addr_str);
    } else {
        printf("Sensor Loop: No link-local address available\n");
    }
}

/**
 * @brief Resets all energest counters after sending metrics.
 */
static void energest_clear_reg(void) {
    energest_type_set(ENERGEST_TYPE_CPU, 0);
    energest_type_set(ENERGEST_TYPE_LPM, 0);
    energest_type_set(ENERGEST_TYPE_TRANSMIT, 0);
    energest_type_set(ENERGEST_TYPE_LISTEN, 0);
}

/**
 * @brief (Optional) Prints the metrics packet for debugging purposes.
 */
#ifdef DEBUG_PRINT_METRICS_PACKET
static void print_metrics(node_metrics_packet_t *metrics) {
    printf("Packet:\n");
    printf("    number: %d\n", metrics->packet_number);
    printf("    time: %lu\n", metrics->current_time);
    printf("Energy consumption:\n");
    printf("    cpu_energy_mJ = %d\n", metrics->cpu_energy_mJ);
    printf("    lpm_energy_mJ = %d\n", metrics->lpm_energy_mJ);
    printf("    radio_tx_energy_mJ = %d\n", metrics->radio_tx_energy_mJ);
    printf("    radio_rx_energy_mJ = %d\n", metrics->radio_rx_energy_mJ);
    printf("Network:\n");
    printf("    total_sent = %d\n", metrics->total_sent);
    printf("    total_received = %d\n", metrics->total_received);
    printf("    bytes_tx = %d\n", metrics->bytes_tx);
    printf("    bytes_rx = %d\n", metrics->bytes_rx);
    printf("    root_to_node_latency = %lu\n", metrics->from_root_to_node_latency);
}
#endif

/**
 * @brief Fills the metrics packet with current readings.
 *
 * Collects energy counters, traffic statistics, latency, and link quality.
 */
static void fill_node_metrics_packet(node_metrics_packet_t *metrics) {
    float cpu_time = to_seconds(energest_type_time(ENERGEST_TYPE_CPU));
    float lpm_time = to_seconds(energest_type_time(ENERGEST_TYPE_LPM));
    float tx_time  = to_seconds(energest_type_time(ENERGEST_TYPE_TRANSMIT));
    float rx_time  = to_seconds(energest_type_time(ENERGEST_TYPE_LISTEN));

#ifdef DEBUG_ENERGY_TIME_IN_SECONDS
    printf("Energy time (s): cpu=%.2f lpm=%.2f tx=%.2f rx=%.2f\n",
           cpu_time, lpm_time, tx_time, rx_time);
#endif

    metrics->cpu_energy_mJ       = CPU_POWER_ACTIVE * cpu_time;
    metrics->lpm_energy_mJ       = LPM_POWER * lpm_time;
    metrics->radio_tx_energy_mJ  = RADIO_TX_POWER * tx_time;
    metrics->radio_rx_energy_mJ  = RADIO_RX_POWER * rx_time;

    metrics->total_sent     = total_sent;
    metrics->total_received = total_received;
    metrics->bytes_tx       = bytes_tx;
    metrics->bytes_rx       = bytes_rx;

    metrics->packet_number = total_sent - 1;
    metrics->current_time = TICKS_TO_MS(NOW_TICKS());
    metrics->from_root_to_node_latency = root_to_node_latency;

    metrics->last_rssi = last_rssi;
    metrics->last_lqi  = last_lqi;
}

//------------------------ UDP Receive Callback -------------------------------

/**
 * @brief Callback triggered when a UDP packet is received.
 *
 * - Updates RSSI, LQI, and RX counters.
 * - If the packet is a PING, sends a PONG response.
 */
static void udp_rx_callback(struct simple_udp_connection *c,
                            const uip_ipaddr_t *sender_addr,
                            uint16_t sender_port,
                            const uip_ipaddr_t *receiver_addr,
                            uint16_t receiver_port,
                            const uint8_t *data,
                            uint16_t datalen) 
{
#ifdef DEBUG_RX_CALLBACK
    char addr_str[UIPLIB_IPV6_MAX_STR_LEN];
    uiplib_ipaddr_snprint(addr_str, sizeof(addr_str), sender_addr);
    printf("UDP RX Sender = %s\n", addr_str);
    printf("Received bytes = %d\n", datalen);
#endif

    total_received++;
    bytes_rx = datalen;

    NETSTACK_RADIO.get_value(RADIO_PARAM_LAST_RSSI, &last_rssi);
    NETSTACK_RADIO.get_value(RADIO_PARAM_LAST_LINK_QUALITY, &last_lqi);

    uint64_t now = TICKS_TO_MS(NOW_TICKS());

    if (datalen == sizeof(ping_packet_t)) {
        ping_packet_t *pkt = (ping_packet_t *)data;
        root_to_node_latency = now - pkt->send_timestamp;
        // Echo the packet back as a PONG
        simple_udp_sendto(&udp_conn, pkt, sizeof(ping_packet_t), sender_addr);
    }
}

//------------------------ Main Process ---------------------------------------

/**
 * @process udp_client_process
 * Main process responsible for:
 *  - registering UDP socket
 *  - checking connectivity with the root
 *  - periodically sending metrics packets
 */
PROCESS(udp_client_process, "UDP client");
AUTOSTART_PROCESSES(&udp_client_process);

PROCESS_THREAD(udp_client_process, ev, data)
{
    static struct etimer periodic_timer;
    uip_ipaddr_t dest_ipaddr;
    static node_metrics_packet_t metrics;

    PROCESS_BEGIN();

    simple_udp_register(&udp_conn, UDP_CLIENT_PORT, NULL, UDP_SERVER_PORT, udp_rx_callback);
    etimer_set(&periodic_timer, BASE_INTERVAL + JITTER);

    NETSTACK_MAC.on();

    while(1) {
        PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&periodic_timer));

        if (NETSTACK_ROUTING.node_is_reachable() &&
            NETSTACK_ROUTING.get_root_ipaddr(&dest_ipaddr)) {

            total_sent++;
            bytes_tx = sizeof(metrics);

            energest_flush();
            fill_node_metrics_packet(&metrics);
            energest_clear_reg();

            simple_udp_sendto(&udp_conn, &metrics, bytes_tx, &dest_ipaddr);
            print_own_link_local();

#ifdef DEBUG_PRINT_METRICS_PACKET
            print_metrics(&metrics);
#endif

        } else {
            printf("Not reachable yet\n");
        }

        etimer_reset(&periodic_timer);
    }

    PROCESS_END();
}
