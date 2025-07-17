# see https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
# 0: charged hadrons
# 1: electrons
# 2: muons
# 3: neutral hadrons
# 4: photons
# 5: residual
# -1: neutrinos

pdgid_class_dict = {
    -211: 0,
    211: 0,  # pi+-
    -213: 0,
    213: 0,  # rho+-
    -221: 0,
    221: 0,  # eta+-
    -223: 0,
    223: 0,  # omega(782)
    -321: 0,
    321: 0,  # kaon+-
    -323: 0,
    323: 0,  # K*+-
    -331: 3,
    331: 3,  # eta'(958)
    -333: 3,
    333: 3,  # phi(1020)
    -411: 0,
    411: 0,  # D+-
    -413: 0,
    413: 0,  # D*(2010)+-
    -423: 3,
    423: 3,  # D*(2007)0
    -431: 0,
    431: 0,  # D_s+-
    -433: 0,
    433: 0,  # D_s*+-
    -511: 3,
    511: 3,  # B0
    -521: 0,
    521: 0,  # B+-
    -523: 0,
    523: 0,  # B*+-
    -531: 3,
    531: 3,  # Bs0
    -541: 0,
    541: 0,  # B_c+-
    -1114: 0,
    1114: 0,  # delta+-
    -2114: 0,
    2114: 0,  # delta0
    -2212: 0,
    2212: 0,  # proton
    -3112: 0,
    3112: 0,  # sigma-
    -3312: 0,
    3312: 0,  # xi+-
    -3222: 0,
    3222: 0,  # sigma+
    -3334: 0,
    3334: 0,  # omega
    -4122: 0,
    4122: 0,  # lambda_c+
    -4132: 3,
    4132: 3,  # xi_c0
    -4232: 0,
    4232: 0,  # xi_c+-
    -4312: 0,
    4312: 0,  # xi'_c0
    -4322: 0,
    4322: 0,  # xi'_c+-
    -4324: 0,
    4324: 0,  # xi*c+-
    -4332: 3,
    4332: 3,  # omega_c0
    -4334: 3,
    4334: 3,  # omega*_c0
    -5112: 0,
    5112: 0,  # lambdab-
    -5122: 3,
    5122: 3,  # lambdab0
    -5132: 0,
    5132: 0,  # xib-
    -5232: 3,
    5232: 3,  # xi0_b
    -5332: 0,
    5332: 0,  # omega_b-
    -11: 1,
    11: 1,  # e
    -13: 2,
    13: 2,  # mu
    -15: 0,
    15: 0,  # tau (calling it charged hadron)
    -111: 3,
    111: 3,  # pi0
    113: 3,  # rho0
    130: 3,  # K0L
    310: 3,  # K0S
    -311: 3,
    311: 3,  # K0
    -313: 3,
    313: 3,  # K*0
    -421: 3,
    421: 3,  # D0
    -2112: 3,
    2112: 3,  # neutrons
    -3122: 3,
    3122: 3,  # lambda
    -3322: 3,
    3322: 3,  # xi0
    22: 4,  # photon
    1000010020: 0,  # deuteron
    1000010030: 0,  # triton
    1000010040: 0,  # alpha
    1000020030: 0,  # He3
    1000020040: 0,  # He4
    1000030040: 0,  # Li6
    1000030050: 0,  # Li7
    1000020060: 0,  # C6
    1000020070: 0,  # C7
    1000020080: 0,  # O8
    1000010048: 0,  # no clue what this is
    1000020032: 0,  # no clue what this is
    -999: 5,  # residual
    -12: -1,
    12: -1,  # nu_e
    -14: -1,
    14: -1,  # nu_mu
    -16: -1,
    16: -1,  # nu_tau
}


# see https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
# 0: charged hadrons
# 1: electrons
# 2: muons
# 3: neutral hadrons
# 4: photons
# 5: residual
# -1: neutrinos

class_mass_dict = {
    0: 0.2760,  # ch had
    1: 0.00051,  # e
    2: 0.10566,  # mu
    3: 0.76419,  # neut had
    4: 0.0,  # gamma
    5: 0.0,  # residual
}
