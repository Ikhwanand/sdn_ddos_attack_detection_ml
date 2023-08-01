from pox.core import core
import pox.openflow.libopenflow_01 as of
from datetime import datetime
from pox.lib import hub

class CollectTrainingStatsApp(object):
    def __init__(self):
        self.logger = core.getLogger()
        self.datapaths = {}
        core.openflow.addListeners(self)
        self.monitor_thread = hub.spawn(self.monitor)

    def _handle_ConnectionUp(self, event):
        datapath = event.connection
        if datapath.dpid not in self.datapaths:
            self.logger.debug('register datapath: %s', datapath.dpid)
            self.datapaths[datapath.dpid] = datapath

    def _handle_ConnectionDown(self, event):
        datapath = event.connection
        if datapath.dpid in self.datapaths:
            self.logger.debug('unregister datapath: %s', datapath.dpid)
            self.datapaths[datapath.dpid] = datapath

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
        timestamp = datetime.now().timestamp()
        icmp_code = -1
        icmp_type = -1
        tp_src = 0
        tp_dst = 0
        file0 = open('FlowStatsFile.csv', 'a+')
        for stat in event.stats:
            if stat.priority == 1:
                ip_src = stat.match['ipv4_src']
                ip_dst = stat.match['ipv4_dst']
                ip_proto = stat.match['ip_proto']
                if stat.match['ip_proto'] == 1:
                    icmp_code = stat.match['icmpv4_code']
                    icmp_type = stat.match['icmpv4_type']
                elif stat.match['ip_proto'] == 6:
                    tp_src = stat.match['tcp_src']
                    tp_dst = stat.match['tcp_dst']
                elif stat.match['ip_proto'] == 17:
                    tp_src = stat.match['udp_src']
                    tp_dst = stat.match['udp_dst']
                flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}"
                try:
                    packet_count_per_second = stat.packet_count / stat.duration_sec
                    packet_count_per_nsecond = stat.packet_count / stat.duration_nsec
                except:
                    packet_count_per_second = 0
                    packet_count_per_nsecond = 0
                try:
                    byte_count_per_second = stat.byte_count / stat.duration_sec
                    byte_count_per_nsecond = stat.byte_count / stat.duration_nsec
                except:
                    byte_count_per_second = 0
                    byte_count_per_nsecond = 0
                file0.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(timestamp, event.connection.dpid, flow_id, ip_src, tp_src, ip_dst, tp_dst, ip_proto, 
                                                                                                         icmp_code, icmp_type, stat.duration_sec, stat.duration_nsec, stat.idle_timeout, stat.hard_timeout, 
                                                                                                         stat.flags, stat.packet_count, stat.byte_count, packet_count_per_second, packet_count_per_nsecond, 
                                                                                                         byte_count_per_second, byte_count_per_nsecond, 1))
        file0.close()