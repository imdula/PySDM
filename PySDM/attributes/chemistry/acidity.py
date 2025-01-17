from PySDM.attributes.impl.intensive_attribute import DerivedAttribute
from PySDM.physics.aqueous_chemistry.support import AQUEOUS_COMPOUNDS
from PySDM.physics import constants as const


class Acidity(DerivedAttribute):
    def __init__(self, builder):
        self.conc = {}
        for key, val in AQUEOUS_COMPOUNDS.items():
            if len(val) > 1:
                self.conc[key] = builder.get_attribute('conc_' + key)
        super().__init__(builder, name='pH', dependencies=self.conc.values())
        self.environment = builder.particulator.environment
        self.cell_id = builder.get_attribute('cell id')
        self.particles = builder.particulator

    def allocate(self, idx):
        super().allocate(idx)
        self.data[:] = const.pH_w

    def recalculate(self):
        dynamic = self.particles.dynamics['AqueousChemistry']

        cell_id = self.cell_id.get()
        N_mIII = self.conc["N_mIII"].get()
        N_V = self.conc["N_V"].get()
        C_IV = self.conc["C_IV"].get()
        S_IV = self.conc["S_IV"].get()
        S_VI = self.conc["S_VI"].get()

        self.particulator.bck.equilibrate_H(
            dynamic.equilibrium_consts, cell_id, N_mIII, N_V, C_IV, S_IV, S_VI,
            dynamic.do_chemistry_flag, self.data,
            H_min=dynamic.pH_H_min,
            H_max=dynamic.pH_H_max,
            ionic_strength_threshold=dynamic.ionic_strength_threshold,
            rtol=dynamic.pH_rtol
        )
