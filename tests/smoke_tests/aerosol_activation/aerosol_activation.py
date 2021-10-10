import numpy as np
from matplotlib import pyplot
from PySDM.physics import si, spectra
from PySDM.initialisation import spectral_sampling, multiplicities, r_wet_init
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM import Builder, products

env = Parcel(
    dt = .25 * si.s,
    mass_of_dry_air = 1e3 * si.kg,
    p0 = 1122 * si.hPa,
    q0 = 20 * si.g / si.kg,
    T0 = 300 * si.K,
    w = 2.5 * si.m / si.s
)
kappa = .5 * si.dimensionless
cloud_range = (.5 * si.um, 25 * si.um)
output_interval = 4
output_points = 40
n_sd = 256

builder = Builder(backend=CPU(), n_sd=n_sd)
builder.set_environment(env)
builder.add_dynamic(AmbientThermodynamics())
builder.add_dynamic(Condensation())

mode_1 = spectra.Lognormal(norm_factor = 1e4 / si.mg, m_mode = 40.0 * si.nm,  s_geom = 1.4)
mode_2 = spectra.Lognormal(norm_factor = 1e3 / si.mg, m_mode = 150.0 * si.nm, s_geom = 1.6)
r_dry, specific_concentration = spectral_sampling.Logarithmic(
            spectrum=spectra.Sum((mode_1, mode_2)),
            size_range=(10.0 * si.nm, 500.0 * si.nm)
        ).sample(n_sd)
v_dry = builder.formulae.trivia.volume(radius=r_dry)
r_wet = r_wet_init(r_dry, env, kappa * v_dry)

r_bins_edges = np.linspace(0 * si.nm, 500.0 * si.nm, 1001, endpoint=True)

attributes = {
    'n': multiplicities.discretise_n(specific_concentration * env.mass_of_dry_air),
    'dry volume': v_dry,
    'kappa times dry volume': kappa * v_dry,
    'volume': builder.formulae.trivia.volume(radius=r_wet)
}

particulator = builder.build(attributes, products=[
    products.PeakSupersaturation(),
    products.CloudDropletEffectiveRadius(radius_range=cloud_range),
    products.CloudDropletConcentration(radius_range=cloud_range),
    products.WaterMixingRatio(radius_range=cloud_range),
    products.ActivatingRate(),
    products.DeactivatingRate(),
    products.ParcelDisplacement(),
    products.ParticlesWetSizeSpectrum(radius_bins_edges=r_bins_edges)
])

cell_id = 0
output = {product.name: [product.get()[cell_id]] for product in particulator.products.values()}

for step in range(output_points):
    particulator.run(steps=output_interval)
    for product in particulator.products.values():
        output[product.name].append(product.get()[cell_id])

#fig, axs = pyplot.subplots(1, 6, sharey="all")
#for i, (key, product) in enumerate(particulator.products.items()):
#    if key in ["S_max", "r_eff", "n_c_cm3", "ql", "activating_rate", "deactivating_rate"]:
#        axs[i].plot(output[key], output['z'], marker='.')
#        axs[i].set_title(product.name)
#        axs[i].set_xlabel(product.unit)
#        axs[i].grid()
#axs[0].set_ylabel(particulator.products['z'].unit)
#pyplot.savefig('parcel.svg')


print(output["Particles Wet Size Spectrum"])

#r_bins_values = np.array(output["Particles Wet Size Spectrum"]) / env.mass_of_dry_air

#print(r_bins_edges.shape)
#print(r_bins_values.shape)

#fig, ax = pyplot.subplots(1, 1, sharex=True)
#pyplot.step(x=r_bins_edges[:-1] / si.um, y=r_bins_values, where='post', label=f"t = {step}s")
#pyplot.xscale('log')
#pyplot.yscale("log")
#ax.legend(loc='best')
#ax.grid()
#pyplot.tight_layout()
#ax.set_title('Wet radius size distribution')
#ax.set_xlabel("wet radius [um]")
#ax.set_ylabel("dN/dlog_10(r) [1/mg 1/um]")
#fig.subplots_adjust(top=0.88)
#pyplot.legend()
#pyplot.savefig('parcel_size_distr.svg')
