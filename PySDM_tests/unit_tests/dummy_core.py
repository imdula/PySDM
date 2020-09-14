"""
Created at 19.11.2019
"""

from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.core import Core
from .dummy_environment import DummyEnvironment
from PySDM.attributes.droplet.multiplicities import Multiplicities
from PySDM.attributes.cell.cell_id import CellID


class DummyCore(Builder, Core):

    def __init__(self, backend=None, n_sd=0):
        backend = backend or CPU
        Core.__init__(self, n_sd, backend)
        self.core = self
        self.environment = DummyEnvironment()
        self.environment.register(self)
        self.req_attr = {'n': Multiplicities(self), 'cell id': CellID(self)}
        self.state = None
