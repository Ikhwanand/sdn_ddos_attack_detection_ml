from pox.core import core
import pox.openflow.libopenflow_01 as of
from pox.openflow.libopenflow_01 import out_packet_out     
from pox.lib.packet.ethernet import ethernet, ether_types
from pox.lib.packet.ipv4 import ipv4
from pox.lib.packet.icmp import icmp
from pox.lib.packet.tcp import tcp
from pox.lib.packet.udp import udp
from pox.lib import packet

import logging


log = core.getLogger()

class SimpleSwitch(object):

    def __init__(self):
        core.openflow.addListeners(self)
        self.mac_to_port = {}

    def _handle_ConnectionUp(self, event):
        con = event.connection
        match = of.ofp_match()
        actions = [of.ofp_action_output(port=of.OFPP_CONTROLLER)]
        self.add_flow(con, 0, match, actions)

    def add_flow(self, con, priority, match, actions):
        msg = of.ofp_flow_mod()
        msg.priority = priority
        msg.match = match
        msg.actions.extend(actions)
        con.send(msg)
    
    def _packet_in_handler(self, event):
        packet_in_msg = event.parsed

         # Ignore LLDP packets
        if packet_in_msg.type == ethernet.LLDP_TYPE:
            return

        datapath_id_str=event.dpid.toStr()
        in_port_num=event.port


        # Fetching the src and dst MAC addresses from the Ethernet header fields.
        eth_pkt=packet.Packet(packet_in_msg.data).get_protocol(ethernet)   
        src_mac_addr_str=eth_pkt.src.toStr()    
        dst_mac_addr_str=eth_pkt.dst.toStr()    

        # Storing the source MAC address and incoming port information in a dictionary.
        if not (datapath_id_str in self.mac_to_port):
            self.mac_to_port[datapath_id_str] = {}
            self.mac_to_port[datapath_id_str][src_mac_addr_str] = in_port_num

            # Checking if the destination MAC address is already known
            if dst_mac_addr_str in self.mac_to_port[datapath_id_str]:
                out_port_num=self.mac_to_port[datapath_id_str][dst_mac_addr_str]
            else:
                out_port_num=of.OFPP_FLOOD

            # Adding an output action to forward the packet to the appropriate port.
            actions=[of.ofp_action_output(port=out_port_num)]
            data=None
            if event.ofp.buffer_id != -1 and event.ofp.buffer_id is not None:
                # If a buffer ID is provided, use it to retrieve buffered packet data.
                data=event.ofp.data
            elif packet_in_msg.next: 
                data=packet_in_msg.pack()

            # Creating and sending OpenFlow PacketOut message with appropriate parameters.  
            msg=of.ofp_packet_out(data=data,action=actions,
                                  buffer_id=event.ofp.buffer_id,in_port=in_port_num)
            event.connection.send(msg)

# Menjalankan SimpleSwitch ketika POX dijalankan
def launch():
    core.registerNew(SimpleSwitch)