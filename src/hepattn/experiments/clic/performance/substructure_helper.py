import energyflow as ef
import numpy as np

# def compute_substructure_leading(pt, eta, phi, jets, n_procs=1, event_number=None):

#     if event_number is None:
#         event_number = np.arange(len(pt))

#     selected_event_numbers = []; pt_eta_phis = []
#     for pt_i, eta_i, phi_i, j, ev_i in zip(pt, eta, phi, jets, event_number):
#         if len(j) == 0:
#             continue
#         const_idxs = j[0].constituent_idxs
#         pt_eta_phis.append(np.stack([
#             pt_i[const_idxs], eta_i[const_idxs], phi_i[const_idxs]
#         ], axis=-1))
#         selected_event_numbers.append(ev_i)

#     d2_calc = ef.D2(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)
#     c2_calc = ef.C2(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)
#     c3_calc = ef.C3(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)

#     d2 = d2_calc.batch_compute(pt_eta_phis, n_jobs=n_procs)
#     c2 = c2_calc.batch_compute(pt_eta_phis, n_jobs=n_procs)
#     c3 = c3_calc.batch_compute(pt_eta_phis, n_jobs=n_procs)

#     return np.array(d2), np.array(c2), np.array(c3), np.array(selected_event_numbers)


def compute_substructure_leading(pt, eta, phi, mass, jets, n_procs=1, event_number=None):
    if event_number is None:
        event_number = np.arange(len(pt))

    selected_event_numbers = []
    pt_eta_phi_massess = []
    for pt_i, eta_i, phi_i, mass_i, j, ev_i in zip(pt, eta, phi, mass, jets, event_number, strict=False):
        if len(j) == 0:
            continue
        const_idxs = j[0].constituent_idxs
        pt_eta_phi_massess.append(np.stack([pt_i[const_idxs], eta_i[const_idxs], phi_i[const_idxs], mass_i[const_idxs]], axis=-1))
        selected_event_numbers.append(ev_i)

    d2_calc = ef.D2(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)
    c2_calc = ef.C2(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)
    c3_calc = ef.C3(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)

    d2 = d2_calc.batch_compute(pt_eta_phi_massess, n_jobs=n_procs)
    c2 = c2_calc.batch_compute(pt_eta_phi_massess, n_jobs=n_procs)
    c3 = c3_calc.batch_compute(pt_eta_phi_massess, n_jobs=n_procs)

    return np.array(d2), np.array(c2), np.array(c3), np.array(selected_event_numbers)
