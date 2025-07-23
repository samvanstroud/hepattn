import fastjet as fj
import numpy as np
import vector as vec
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

from .cheap_jet import CheapJet


class JetHelper:
    def __init__(self, radius, algo="antikt"):
        self.radius = radius
        self.algo = algo
        assert algo in {"antikt", "genkt"}, f"Jet algorithm {algo} not implemented!"

        if self.algo == "genkt":
            self.jetdef = fj.JetDefinition(fj.ee_genkt_algorithm, self.radius, -1.0)
        elif self.algo == "antikt":
            self.jetdef = fj.JetDefinition(fj.antikt_algorithm, self.radius)

        print("Jet clustering algorithm: ", algo)
        print("Jet clustering radius: ", self.radius)

    def getvectors(self, pt_array, eta_array, phi_array, fourth_array, fourth_name):
        # fourth_name can be either 'mass' or 'E'
        return vec.array({"pt": pt_array, "eta": eta_array, "phi": phi_array, fourth_name: fourth_array})

    def getclustersequence(self, particles, user_indices=None):
        pj_array = []
        for i, part in enumerate(particles):
            pj = fj.PseudoJet(part.px.item(), part.py.item(), part.pz.item(), part.E.item())
            if user_indices is not None:
                pj.set_user_index(user_indices[i])
            else:
                pj.set_user_index(i)
            pj_array.append(pj)
        return fj.ClusterSequence(pj_array, self.jetdef)

    def getconstituentmap(self, cs, ptmin=2):
        jet_map = {}

        jets = cs.inclusive_jets(ptmin)
        jets = fj.sorted_by_pt(jets)

        for i, jet in enumerate(jets):
            for constit in jet.constituents():
                constit_idx = constit.user_index()
                if constit_idx in jet_map:
                    print(f"WARNING: constituent {constit_idx} already assigned to jet {jet_map[constit_idx]}! Reassigning to jet {i}...")
                jet_map[constit_idx] = [i, jet.pt(), jet.eta(), jet.phi()]

        return jet_map

    def getptsortedjets(self, four_vectors, pt_min=8, n_const_min=2, eta_max=2.5, get_constituent_idxs=False):
        cs = self.getclustersequence(four_vectors)
        jets = cs.inclusive_jets(ptmin=pt_min)
        jets = fj.sorted_by_pt(jets)
        jets = [j for j in jets if len(j.constituents()) >= n_const_min]
        jets = [j for j in jets if abs(j.eta()) < eta_max]
        n_constituents = [len(j.constituents()) for j in jets]
        constituent_idxs = [None for j in jets]
        if get_constituent_idxs:
            constituent_idxs = [np.array([constit.user_index() for constit in j.constituents()]) for j in jets]
        return zip(jets, n_constituents, constituent_idxs, strict=False)

    def getptsortedcheapjets(self, four_vectors, pt_min=8, n_const_min=2, eta_max=2.5, store_constituent_idxs=False):
        return [
            CheapJet(j, n_const=nc, constituent_idxs=const_idxs)
            for (j, nc, const_idxs) in self.getptsortedjets(four_vectors, pt_min, n_const_min, eta_max, get_constituent_idxs=store_constituent_idxs)
        ]

    def compute_jets(self, pts, etas, phis, fourths, fourth_name, store_constituent_idxs=False):
        # fourth_name can be either 'mass' or 'E'
        n_events = len(pts)
        jets = []
        for ev_i in tqdm(range(n_events), desc="Computing jets..."):
            four_vec = self.getvectors(pts[ev_i], etas[ev_i], phis[ev_i], fourths[ev_i], fourth_name)
            jets.append(self.getptsortedcheapjets(four_vec, store_constituent_idxs=store_constituent_idxs))
        return jets


# multiprocessed version
def compute_jet_multiproc(kwargs, pts, etas, phis, fourths, fourth_name, store_constituent_idxs=False):
    jet_helper_obj = JetHelper(**kwargs)
    return jet_helper_obj.compute_jets(pts, etas, phis, fourths, fourth_name, store_constituent_idxs)


def compute_jets(jet_helper_obj, pts, etas, phis, fourths, fourth_name, n_procs=0, store_constituent_idxs=False):
    # fourth_name can be either 'mass' or 'E'
    n_events = len(pts)
    if n_procs > 0:
        with Pool(n_procs) as pool:
            start_stops = np.linspace(0, n_events, n_procs + 1, dtype=int)
            starts, ends = start_stops[:-1], start_stops[1:]
            results = []
            for _i, (start, end) in enumerate(zip(starts, ends, strict=False)):
                result = pool.apipe(
                    compute_jet_multiproc,
                    {"radius": jet_helper_obj.radius, "algo": jet_helper_obj.algo},
                    *(pts[start:end], etas[start:end], phis[start:end], fourths[start:end], fourth_name, store_constituent_idxs),
                )
                results.append(result)
            jets = [jet for sublist in results for jet in sublist.get()]

    else:
        jets = jet_helper_obj.compute_jets(pts, etas, phis, fourths, fourth_name, store_constituent_idxs)

    return jets
