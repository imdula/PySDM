import numpy as np
from matplotlib import pyplot
from PySDM.physics import si, spectra
from PySDM.initialisation import spectral_sampling, multiplicities, r_wet_init
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM import Builder, products

env = Parcel(
    dt = .1 * si.s,
    mass_of_dry_air = 1e3 * si.kg,
    p0 = 1122 * si.hPa,
    q0 = 20 * si.g / si.kg,
    T0 = 300 * si.K,
    w = 2.5 * si.m / si.s
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

# define first mode
r_1 =30.0 * si.nm
mode_1 = spectra.Lognormal(norm_factor = 1e4 / si.mg, m_mode = 30.0 * si.nm,  s_geom = 1.6)
#initialize some arrays:
T_sat = []
p_sat = []
N_act = []
# Define modes that start close to mode 1 and get further away
# loop through each of these r_factors that multiply the mode 2 radius by this factor
r_factors = np.linspace(1.1, 10, 20, endpoint = False)

for r_factor in r_factors:
    r_2 = r_1 * r_factor
    mode_2 = spectra.Lognormal(norm_factor = 2e3 / si.mg, m_mode = r_2, s_geom = 1.2)
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
    
    # Determine supersaturation conditions for this sim:
    S =  np.array(output['S_max']).astype(int)
    print(len(S))
    print(S)
    print(S[0][0])
    #trying some stuff to get rid of errors:
    T = np.array(output['T_env']).astype(int)
    print(T)
    print(T[0][0])
    p = np.array(output['p_env']).astype(int)
    for i in range(len(S)-1):
        saturation = S[i][0]
        if saturation < 0 or np.isnan(saturation)== True:
            continue
        else:
            print(T[i][0])
            T_sat.append((T[i][0] + T[i-1][0]) / 2)
            p_sat.append((p[i][0] + p[i-1][0]) / 2)
            break
    # Determine total activated particles:
    total_act = np.sum(output["activating_rate"]) * .1
    total_deact = np.sum(output["deactivating_rate"]) * .1
    N_act.append((total_act-total_deact )/ 2000)

print(T_sat)
print('-------------------')
print(p_sat)
print('-------------------')
print(N_act)
print('-------------------')
print(r_factors)


