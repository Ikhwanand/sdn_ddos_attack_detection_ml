from pox.core import core
import pox.openflow.libopenflow_01 as of
from datetime import datetime
from pox.lib import hub 

class CollectTrainingStatsApp(object):
    def __init__(self):
        self.datapaths = {}
        self.logger = core.getLogger()
        core.openflow.addListeners(self)

        with open('FlowStatsfile.csv', 'w') as file0:
            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond,label\n')

    def _handle_ConnectionUp(self, event):
        datapath = event.connection
        if datapath.dpid not in self.datapaths:
            self.logger.debug('register datapath: %s', datapath.dpid)
            self.datapaths[datapath.dpid] = datapath

    def _handle_ConnectionDown(self, event):
        datapath = event.connection
        if datapath.dpid in self.datapaths:
            self.logger.debug('unregister datapath: %s', datapath.dpid)
            del self.datapaths[datapath.dpid]

    def monitor(self):
        while True:
            for dp in self.datapaths.values():
                self.request_stats(dp)
            hub.sleep(10)

    def request_stats(self, datapath):
        self.logger.debug('send stats request: %s', datapath.dpid)
        req = of.ofp_stats_request()
        req.body = of.ofp_flow_stats_request()
        datapath.send(req)

    def _handle_FlowStatsReceived(self, event):
        datapath = event.connection
        timestamp = datetime.now().timestamp()
        icmp_code = -1
        icmp_type = -1
        tp_src = 0
        tp_dst = 0

        with open('FlowStatsfile.csv', 'a+') as file0:
            for stat in event.stats:
                if stat.priority == 1:
                    ip_src = stat.match.nw_src
                    ip_dst = stat.match.nw_dst
                    ip_proto = stat.match.nw_proto

                    if stat.match.tp:
                        tp_src = stat.match.tp.src_port
                        tp_dst = stat.match.tp.dst_port
                    if stat.match.dl_type == 0x0800 and stat.match.nw_proto == 1:
                        icmp_code = stat.match.tp.icmp_code
                        icmp_type = stat.match.tp.icmp_type
                    
                    flow_duration_sec = stat.duration_sec
                    flow_duration_nsec = stat.duration_nsec
                    idle_timeout = stat.idle_timeout
                    hard_timeout = stat.hard_timeout
                    flags = stat.flags
                    packet_count = stat.packet_count
                    byte_count = stat.byte_count
                    packet_count_per_second = packet_count / (flow_duration_sec + (flow_duration_nsec / 1e9))
                    packet_count_per_nsecond = packet_count / (flow_duration_sec * 1e9 + flow_duration_nsec)
                    byte_count_per_second = byte_count / (flow_duration_sec + (flow_duration_nsec / 1e9))
                    byte_count_per_nsecond = byte_count / (flow_duration_sec * 1e9 + flow_duration_nsec)
                    label = ""
                    
                    file0.write(f'{timestamp},{datapath.dpid},{stat.match.flow_id},{ip_src},{tp_src},{ip_dst},{tp_dst},{ip_proto},{icmp_code},{icmp_type},{flow_duration_sec},{flow_duration_nsec},{idle_timeout},{hard_timeout},{flags},{packet_count},{byte_count},{packet_count_per_second},{packet_count_per_nsecond},{byte_count_per_second},{byte_count_per_nsecond},{label}\n')