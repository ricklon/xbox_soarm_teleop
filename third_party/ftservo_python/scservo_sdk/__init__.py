#!/usr/bin/env python

from . import group_sync_read as _group_sync_read
from . import group_sync_write as _group_sync_write
from .protocol_packet_handler import protocol_packet_handler as _protocol_packet_handler
from .port_handler import PortHandler
from .protocol_packet_handler import *  # noqa: F403
from .sms_sts import *  # noqa: F403
from .scscl import *  # noqa: F403
from .hls import *  # noqa: F403


def SCS_LOBYTE(value: int) -> int:
    return value & 0xFF


def SCS_HIBYTE(value: int) -> int:
    return (value >> 8) & 0xFF


def SCS_LOWORD(value: int) -> int:
    return value & 0xFFFF


def SCS_HIWORD(value: int) -> int:
    return (value >> 16) & 0xFFFF


class PacketHandler:
    """Adapter to match LeRobot's expected scservo_sdk PacketHandler API."""

    def __init__(self, protocol_version: int = 0):
        self.protocol_version = protocol_version
        self._handler = None

    def _ensure_handler(self, port_handler):
        if self._handler is None or self._handler.portHandler is not port_handler:
            protocol_end = 1 if self.protocol_version == 1 else 0
            self._handler = _protocol_packet_handler(port_handler, protocol_end)
        return self._handler

    def txPacket(self, port_handler, txpacket):
        return self._ensure_handler(port_handler).txPacket(txpacket)

    def rxPacket(self, port_handler):
        return self._ensure_handler(port_handler).rxPacket()

    def ping(self, port_handler, scs_id):  # noqa: N802
        return self._ensure_handler(port_handler).ping(scs_id)

    def read1ByteTxRx(self, port_handler, scs_id, address):  # noqa: N802
        return self._ensure_handler(port_handler).read1ByteTxRx(scs_id, address)

    def read2ByteTxRx(self, port_handler, scs_id, address):  # noqa: N802
        return self._ensure_handler(port_handler).read2ByteTxRx(scs_id, address)

    def read4ByteTxRx(self, port_handler, scs_id, address):  # noqa: N802
        return self._ensure_handler(port_handler).read4ByteTxRx(scs_id, address)

    def writeTxRx(self, port_handler, scs_id, address, length, data):  # noqa: N802
        return self._ensure_handler(port_handler).writeTxRx(scs_id, address, length, data)

    def getTxRxResult(self, result):  # noqa: N802
        if self._handler is None:
            handler = _protocol_packet_handler(PortHandler(""), 0)
            return handler.getTxRxResult(result)
        return self._handler.getTxRxResult(result)

    def getRxPacketError(self, error):  # noqa: N802
        if self._handler is None:
            handler = _protocol_packet_handler(PortHandler(""), 0)
            return handler.getRxPacketError(error)
        return self._handler.getRxPacketError(error)


class GroupSyncRead:
    """Adapter to match LeRobot's expected scservo_sdk GroupSyncRead API."""

    def __init__(self, port_handler, packet_handler, start_address, data_length):
        self._ph = packet_handler._ensure_handler(port_handler)
        self._inner = _group_sync_read.GroupSyncRead(self._ph, start_address, data_length)

    @property
    def start_address(self):
        return self._inner.start_address

    @start_address.setter
    def start_address(self, value):
        self._inner.start_address = value

    @property
    def data_length(self):
        return self._inner.data_length

    @data_length.setter
    def data_length(self, value):
        self._inner.data_length = value

    def addParam(self, scs_id):
        return self._inner.addParam(scs_id)

    def removeParam(self, scs_id):
        return self._inner.removeParam(scs_id)

    def clearParam(self):
        return self._inner.clearParam()

    def txPacket(self):
        return self._inner.txPacket()

    def rxPacket(self):
        return self._inner.rxPacket()

    def txRxPacket(self):
        return self._inner.txRxPacket()

    def isAvailable(self, scs_id, address, length):  # noqa: N802
        return self._inner.isAvailable(scs_id, address, length)

    def getData(self, scs_id, address, length):  # noqa: N802
        return self._inner.getData(scs_id, address, length)


class GroupSyncWrite:
    """Adapter to match LeRobot's expected scservo_sdk GroupSyncWrite API."""

    def __init__(self, port_handler, packet_handler, start_address, data_length):
        self._ph = packet_handler._ensure_handler(port_handler)
        self._inner = _group_sync_write.GroupSyncWrite(self._ph, start_address, data_length)

    @property
    def start_address(self):
        return self._inner.start_address

    @start_address.setter
    def start_address(self, value):
        self._inner.start_address = value

    @property
    def data_length(self):
        return self._inner.data_length

    @data_length.setter
    def data_length(self, value):
        self._inner.data_length = value

    def addParam(self, scs_id, data):
        return self._inner.addParam(scs_id, data)

    def changeParam(self, scs_id, data):
        return self._inner.changeParam(scs_id, data)

    def removeParam(self, scs_id):
        return self._inner.removeParam(scs_id)

    def clearParam(self):
        return self._inner.clearParam()

    def txPacket(self):
        return self._inner.txPacket()
