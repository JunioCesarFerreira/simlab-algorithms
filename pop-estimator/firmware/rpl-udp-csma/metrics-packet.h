/**
 * @file    metrics-packet.h
 * @brief   Data structures used for exchanging performance metrics between root and node in a Contiki-NG WSN.
 *
 * This header defines the packet formats for:
 * - `ping_packet_t`: Sent from the root node to initiate a round-trip latency measurement (ping).
 * - `node_metrics_packet_t`: Sent from sensor nodes to report energy usage, traffic statistics, and connectivity data.
 */

#ifndef METRICS_PACKET_H_
#define METRICS_PACKET_H_

#include <stdint.h>

/**
 * @struct node_metrics_packet_t
 * @brief Metrics report packet sent from a node to the root.
 *
 * This packet contains execution and communication metrics that are used
 * to analyze the performance and energy profile of each node.
 */
typedef struct {
    /** Sequence number of the packet */
    unsigned int packet_number;

    /** Timestamp at the moment of packet generation (in uptime ticks) */
    unsigned long int current_time;

    /** Energy consumed by the CPU in millijoules (mJ) */
    unsigned int cpu_energy_mJ;

    /** Energy consumed in low-power mode (LPM) in millijoules (mJ) */
    unsigned int lpm_energy_mJ;

    /** Energy consumed by radio transmission (TX) in millijoules (mJ) */
    unsigned int radio_tx_energy_mJ;

    /** Energy consumed by radio reception (RX) in millijoules (mJ) */
    unsigned int radio_rx_energy_mJ;

    /** Total number of packets sent by the node */
    unsigned int total_sent;

    /** Total number of packets received by the node */
    unsigned int total_received;

    /** Total number of bytes transmitted */
    unsigned int bytes_tx;

    /** Total number of bytes received */
    unsigned int bytes_rx;

    /** Measured latency from root to node (in uptime ticks) */
    unsigned long int from_root_to_node_latency;

    /** Last RSSI (Received Signal Strength Indicator) recorded (in dBm) */
    int last_rssi;

    /** Last LQI (Link Quality Indicator) recorded (0–255) */
    int last_lqi;

} __attribute__((packed)) node_metrics_packet_t;
// __attribute__((packed)) ensures the structure is transmitted exactly as laid out in memory,
// without padding, making it consistent across platforms in UDP communication.

/**
 * @struct ping_packet_t
 * @brief Probe packet sent from the root to a node to measure RTT (ping-pong mechanism).
 *
 * The node must echo the same structure back. The root will compute RTT and estimate
 * the network latency from the timestamp in this packet.
 */
typedef struct {
    /** Ping sequence number (monotonically increasing) */
    unsigned int ping_seq;

    /** Timestamp when the ping was sent (in uptime ticks) */
    unsigned long int send_timestamp;

} __attribute__((packed)) ping_packet_t;

#endif /* METRICS_PACKET_H_ */
