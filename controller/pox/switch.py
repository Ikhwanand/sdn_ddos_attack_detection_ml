from pox.core import core
import pox.openflow.libopenflow_01 as of
from pox.lib.packet import packet, ethernet, ipv4, icmp, tcp, udp

import logging


log = core.getLogger()

class SimpleSwitch(object):
    def __init__(self):
        self.mac_to_port = {}

    def _handle_ConnectionUp(self, event):
        dp = event.connection

        # Install the default flow
        msg = of.ofp_flow_mod()
        msg.match = of.ofp_match()
        msg.actions.append(of.ofp_action_output(port=of.OFPP_CONTROLLER))
        dp.send(msg)

    def _handle_PacketIn(self, event):
        packet = event.parsed
        dp = event.connection
        in_port = event.port

        # Ignore LLDP packets
        if packet.type == ethernet.LLDP_TYPE:
            return

        dst = packet.dst
        src = packet.src
        dpid = dp.dpid
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = of.OFPP_FLOOD

        actions = [of.ofp_action_output(port=out_port)]

        if out_port != of.OFPP_FLOOD:
            if packet.type == ethernet.IP_TYPE:
                ip = packet.find_protocol(ipv4.ipv4)
                srcip = ip.srcip
                dstip = ip.dstip
                protocol = ip.protocol

                if protocol == ipv4.ICMP_PROTOCOL:
                    icmp_pkt = packet.find_protocol(icmp.icmp)
                    match = of.ofp_match(dl_type=ethernet.IP_TYPE, nw_src=srcip, nw_dst=dstip, nw_proto=protocol, tp_src=icmp_pkt.type, tp_dst=icmp_pkt.code)
                elif protocol == ipv4.TCP_PROTOCOL:
                    tcp_pkt = packet.find_protocol(tcp.tcp)
                    match = of.ofp_match(dl_type=ethernet.IP_TYPE, nw_src=srcip, nw_dst=dstip, nw_proto=protocol, tp_src=tcp_pkt.srcport, tp_dst=tcp_pkt.dstport)
                elif protocol == ipv4.UDP_PROTOCOL:
                    udp_pkt = packet.find_protocol(udp.udp)
                    match = of.ofp_match(dl_type=ethernet.IP_TYPE, nw_src=srcip, nw_dst=dstip, nw_proto=protocol, tp_src=udp_pkt.srcport, tp_dst=udp_pkt.dstport)
        msg = of.ofp_flow_mod()
        msg.match = match 
        msg.actions = actions 
        dp.send(msg)  


# Menjalankan SimpleSwitch ketika POX dijalankan
def launch():
    core.registerNew(SimpleSwitch)