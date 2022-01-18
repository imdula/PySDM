import numpy as np
from matplotlib import pyplot
from PySDM.physics import si, spectra
from PySDM.initialisation import spectral_sampling, multiplicities, r_wet_init
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM import Builder, products


T = np.linspace(200, 300, 10)
P = np.linspace(900, 1100, 10)
w = np.linspace(.5, 2.0, 5)
sat_w = np.zeros((10,10))
for i in range(len(T)):
    for j in range(len(P)):
        for k in range(len(w)):
            env = Parcel(
                            dt = .1 * si.s,
                            mass_of_dry_air = 1e3 * si.kg,
                            p0 = P[i] * si.hPa,
                            q0 = 20 * si.g / si.kg,
                            T0 = T[i] * si.K,
                            w = w[k] * si.m / si.s
                        )

            kappa = .3 * si.dimensionless # value from kappa kohler
            cloud_range = (.5 * si.um, 25 * si.um)
            output_interval = 10
            output_points = 40
            n_sd = 2024

            builder = Builder(backend=CPU(), n_sd=n_sd)
            builder.set_environment(env)
            builder.add_dynamic(AmbientThermodynamics())
            builder.add_dynamic(Condensation())

            mode_1 = spectra.Lognormal(norm_factor = 1e4 / si.mg, m_mode = 30.0 * si.nm,  s_geom = 1.6)
            mode_2 = spectra.Lognormal(norm_factor = 2e3 / si.mg, m_mode = 120.0 * si.nm, s_geom = 1.2)
            r_dry, specific_concentration = spectral_sampling.Logarithmic(
                        spectrum=spectra.Sum((mode_1, mode_2)),
                        size_range=(10.0 * si.nm, 500.0 * si.nm)
                    ).sample(n_sd)
            v_dry = builder.formulae.trivia.volume(radius=r_dry)
            r_wet = r_wet_init(r_dry, env, kappa * v_dry)

            attributes = {
                'n': multiplicities.discretise_n(specific_concentration * env.mass_of_dry_air),
                'dry volume': v_dry,
                'kappa times dry volume': kappa * v_dry,
                'volume': builder.formulae.trivia.volume(radius=r_wet)
            }

            attributes = {
                'n': multiplicities.discretise_n(specific_concentration * env.mass_of_dry_air),
                'dry volume': v_dry,
                'kappa times dry volume': kappa * v_dry,
                'volume': builder.formulae.trivia.volume(radius=r_wet)
            }

            r_bins_edges = np.linspace(0 * si.nm, 4e3 * si.nm, 101, endpoint=True)
            particulator = builder.build(attributes, products=[
                products.PeakSupersaturation(),
                products.CloudDropletEffectiveRadius(radius_range=cloud_range),
                products.CloudDropletConcentration(radius_range=cloud_range),
                products.WaterMixingRatio(radius_range=cloud_range),
                products.ActivatingRate(),
                products.DeactivatingRate(),
                products.ParcelDisplacement(),
                products.ParticlesWetSizeSpectrum(radius_bins_edges=r_bins_edges),
                products.Temperature(),
                products.Pressure()
            ])



            cell_id = 0
            output = {product.name: [product.get().copy()] for product in particulator.products.values()}

            for step in range(output_points):
                particulator.run(steps=output_interval)
                for product in particulator.products.values():
                    output[product.name].append(product.get().copy())

            S = np.array(output['S_max'])
            if S[-1][0] > 0:
                sat_w[i][j] = w[k]
                break

print(T)
print(P)
print(sat_w)


            



