import gc
from pathlib import Path

import lightning as L
import numpy as np
import torch
import uproot
from lightning import seed_everything
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hepattn.utils.scaling import FeatureScaler


def normalize_phi(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


def do_padding(tensor, max_len):
    x = torch.zeros(max_len, dtype=tensor.dtype, device=tensor.device)
    x[: len(tensor)] = tensor
    return x


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


class CLICDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        inputs: dict,
        targets: dict,
        scale_dict_path: str,
        num_events: int = -1,
        num_objects: int = 150,
        max_nodes: int = 160,
        remove_wrong_idxs: bool = True,
        incidence_cutval: float = 1e-4,
        is_inference: bool = False,
    ):
        super().__init__()

        self.sampling_seed = 42
        np.random.default_rng(self.sampling_seed)
        seed_everything(self.sampling_seed, workers=True)

        self.scaler = FeatureScaler(scale_dict_path)

        # init class dict
        self.init_label_dicts()

        # init variable lists
        self.init_variables_list()

        # input stuff
        self.filepath = filepath
        assert is_valid_file(filepath), f"Invalid file: {filepath}"

        # other properties
        self.inputs = inputs
        self.targets = targets
        self.num_objects = num_objects
        self.max_nodes = max_nodes
        self.remove_wrong_idxs = remove_wrong_idxs
        self.incidence_cutval = incidence_cutval
        self.is_inference = is_inference

        print(f"Loading CLIC dataset from {filepath} with {num_events} samples")
        print(f"Is inference: {self.is_inference}")

        with uproot.open(filepath, num_workers=6) as f:
            tree = f["EventTree"]
            self.num_events = tree.num_entries
            if num_events > 0:
                self.num_events = min(self.num_events, num_events)
            self.load_data(tree)
            gc.collect()

    def load_data(self, tree):
        self.full_data_array = {}

        varlist = self.track_variables + self.topo_variables + self.particle_variables
        arrays = tree.arrays(varlist + self.aux_vars, library="np", entry_stop=self.num_events)
        for var in tqdm(varlist):
            self.full_data_array[var] = arrays[var]
            if "phi" in var and "particle" not in var:
                newvar = var.replace("phi", "sinphi")
                self.full_data_array[newvar] = np.array(
                    [np.sin(x.astype(np.float32)) for x in self.full_data_array[var]],
                    dtype=object,
                )
                newvar = var.replace("phi", "cosphi")
                self.full_data_array[newvar] = np.array(
                    [np.cos(x.astype(np.float32)) for x in self.full_data_array[var]],
                    dtype=object,
                )

            if var == "track_d0":
                self.n_tracks = np.array([len(x) for x in self.full_data_array[var]])
            elif var == "particle_pdgid":
                self.n_particles = np.array([len(x) for x in self.full_data_array[var]])
            if var == "topo_e":
                self.n_topos = np.array([len(x) for x in self.full_data_array[var]])

        self.track_variables = [key for key in self.full_data_array if "track" in key]
        self.topo_variables = [key for key in self.full_data_array if "topo" in key]
        # self.event_number = tree["eventNumber"].array(library="np", entry_stop=self.num_events)
        self.event_number = np.arange(len(self.n_tracks))

        mask = ((self.n_tracks + self.n_topos) < self.max_nodes) & (self.n_particles < self.num_objects)
        print(f"Removing {(~mask).sum()} events with too many nodes or particles")
        print(f"{((self.n_tracks + self.n_topos) >= self.max_nodes).sum()} events with too many nodes")
        print(f"{(self.n_particles >= self.num_objects).sum()} events with too many particles")
        if self.remove_wrong_idxs:
            track_particle_idxs = tree["track_particle_idx"].array(library="np", entry_stop=self.num_events)
            track_particle_idxs_lens = np.array([len(el) for el in track_particle_idxs])
            print(f"Removing {np.sum(track_particle_idxs_lens != self.n_tracks)} events with mismatching track_particle_idx")
            mask &= track_particle_idxs_lens == self.n_tracks

        self.n_tracks = self.n_tracks[mask]
        self.n_topos = self.n_topos[mask]
        self.n_particles = self.n_particles[mask]
        self.event_number = self.event_number[mask]

        for var in self.topo_variables + self.particle_variables + self.track_variables:
            # flatten the arrays
            self.full_data_array[var] = np.concatenate(self.full_data_array[var][mask])
            if var == "particle_pdgid":
                self.particle_class = torch.tensor([self.class_labels[x] for x in self.full_data_array[var]])
            if var in {"track_phi", "track_phi_int", "topo_phi", "particle_phi"}:
                self.full_data_array[var] = normalize_phi(self.full_data_array[var])

        # transform variables and transform to tensors
        for key, val in self.full_data_array.items():
            self.full_data_array[key] = torch.tensor(val)

        self.topo_cumsum = np.cumsum([0, *self.n_topos.tolist()])
        self.track_cumsum = np.cumsum([0, *self.n_tracks.tolist()])
        self.particle_cumsum = np.cumsum([0, *self.n_particles.tolist()])

        # needed for batch sampling
        self.n_nodes = self.n_tracks + self.n_topos

        self.aux_data_array = {}
        for var in self.aux_vars:
            print(f"Loading aux var {var}")
            self.aux_data_array[var] = arrays[var][mask]

        self.num_events = np.sum(mask)
        print(f"Number of events after filtering: {self.num_events}")

        self.neg_contribs = []
        self.fake_TC_count = []

    def __len__(self):
        return int(self.num_events)

    def load_event(self, idx):
        n_topos = self.n_topos[idx]
        n_tracks = self.n_tracks[idx]
        n_particles = self.n_particles[idx]

        n_nodes = n_topos + n_tracks

        topo_start, topo_end = self.topo_cumsum[idx], self.topo_cumsum[idx + 1]
        track_start, track_end = self.track_cumsum[idx], self.track_cumsum[idx + 1]
        particle_start, particle_end = (
            self.particle_cumsum[idx],
            self.particle_cumsum[idx + 1],
        )

        # track features
        track_pt = self.full_data_array["track_pt"][track_start:track_end]
        track_eta = self.full_data_array["track_eta"][track_start:track_end]
        track_phi = self.full_data_array["track_phi"][track_start:track_end]
        track_cosphi = self.full_data_array["track_cosphi"][track_start:track_end]
        track_sinphi = self.full_data_array["track_sinphi"][track_start:track_end]

        track_eta_int = self.full_data_array["track_eta_int"][track_start:track_end]
        track_phi_int = self.full_data_array["track_phi_int"][track_start:track_end]
        track_cosphi_int = self.full_data_array["track_cosphi_int"][track_start:track_end]
        track_sinphi_int = self.full_data_array["track_sinphi_int"][track_start:track_end]

        track_z0 = self.full_data_array["track_z0"][track_start:track_end]
        track_d0 = self.full_data_array["track_d0"][track_start:track_end]
        track_chi2 = self.full_data_array["track_chi2"][track_start:track_end]
        track_ndf = self.full_data_array["track_ndf"][track_start:track_end]
        track_radiusofinnermosthit = self.full_data_array["track_radiusofinnermosthit"][track_start:track_end]
        track_tanlambda = self.full_data_array["track_tanlambda"][track_start:track_end]
        track_omega = self.full_data_array["track_omega"][track_start:track_end]

        # topo features
        topo_e = self.full_data_array["topo_e"][topo_start:topo_end]
        topo_eta = self.full_data_array["topo_eta"][topo_start:topo_end]
        topo_phi = self.full_data_array["topo_phi"][topo_start:topo_end]
        topo_cosphi = self.full_data_array["topo_cosphi"][topo_start:topo_end]
        topo_sinphi = self.full_data_array["topo_sinphi"][topo_start:topo_end]
        topo_rho = self.full_data_array["topo_rho"][topo_start:topo_end]
        topo_sigma_eta = self.full_data_array["topo_sigma_eta"][topo_start:topo_end]
        topo_sigma_phi = self.full_data_array["topo_sigma_phi"][topo_start:topo_end]
        topo_sigma_rho = self.full_data_array["topo_sigma_rho"][topo_start:topo_end]
        topo_energy_ecal = self.full_data_array["topo_energy_ecal"][topo_start:topo_end]
        topo_energy_hcal = self.full_data_array["topo_energy_hcal"][topo_start:topo_end]
        topo_energy_other = self.full_data_array["topo_energy_other"][topo_start:topo_end]
        topo_em_frac = topo_energy_ecal / (topo_energy_ecal + topo_energy_hcal + topo_energy_other)

        node_features = {
            # common features
            "pt": torch.cat(
                [
                    self.scaler.transforms["pt"].transform(track_pt),
                    torch.zeros(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
            "eta": torch.cat(
                [
                    self.scaler.transforms["eta"].transform(track_eta),
                    self.scaler.transforms["eta"].transform(topo_eta),
                ],
                -1,
            ),
            "phi": torch.cat(
                [
                    track_phi,
                    topo_phi,
                ],
                -1,
            ),
            "cosphi": torch.cat([track_cosphi, topo_phi], -1),
            "sinphi": torch.cat([track_sinphi, topo_phi], -1),
            # interaction features
            "eta_int": torch.cat(
                [
                    self.scaler.transforms["eta"].transform(track_eta_int),
                    torch.zeros(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
            "phi_int": torch.cat([track_phi_int, torch.zeros(n_topos, dtype=torch.float32)], -1),
            "cosphi_int": torch.cat([track_cosphi_int, torch.zeros(n_topos, dtype=torch.float32)], -1),
            "sinphi_int": torch.cat([track_sinphi_int, torch.zeros(n_topos, dtype=torch.float32)], -1),
            # track features
            "z0": torch.cat(
                [
                    self.scaler.transforms["z0"].transform(track_z0),
                    torch.zeros(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
            "d0": torch.cat(
                [
                    self.scaler.transforms["d0"].transform(track_d0),
                    torch.zeros(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
            "chi2": torch.cat(
                [
                    self.scaler.transforms["chi2"].transform(track_chi2),
                    torch.zeros(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
            "ndf": torch.cat(
                [
                    self.scaler.transforms["ndf"].transform(track_ndf),
                    torch.zeros(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
            "radiusofinnermosthit": torch.cat(
                [
                    self.scaler.transforms["radiusofinnermosthit"].transform(track_radiusofinnermosthit),
                    torch.zeros(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
            "tanlambda": torch.cat(
                [
                    self.scaler.transforms["tanlambda"].transform(track_tanlambda),
                    torch.zeros(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
            "omega": torch.cat(
                [
                    self.scaler.transforms["omega"].transform(track_omega),
                    torch.zeros(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
            # topo features
            "e": torch.cat(
                [
                    torch.zeros(n_tracks, dtype=torch.float32),
                    self.scaler.transforms["e"].transform(topo_e),
                ],
                -1,
            ),
            "rho": torch.cat(
                [
                    torch.zeros(n_tracks, dtype=torch.float32),
                    self.scaler.transforms["rho"].transform(topo_rho),
                ],
                -1,
            ),
            "sigma_eta": torch.cat(
                [
                    torch.zeros(n_tracks, dtype=torch.float32),
                    self.scaler.transforms["sigma_eta"].transform(topo_sigma_eta),
                ],
                -1,
            ),
            "sigma_phi": torch.cat(
                [
                    torch.zeros(n_tracks, dtype=torch.float32),
                    self.scaler.transforms["sigma_phi"].transform(topo_sigma_phi),
                ],
                -1,
            ),
            "sigma_rho": torch.cat(
                [
                    torch.zeros(n_tracks, dtype=torch.float32),
                    self.scaler.transforms["sigma_rho"].transform(topo_sigma_rho),
                ],
                -1,
            ),
            "energy_ecal": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), topo_energy_ecal], -1),
            "energy_hcal": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), topo_energy_hcal], -1),
            "energy_other": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), topo_energy_other], -1),
            "em_frac": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), topo_em_frac], -1),
            # flags
            "is_track": torch.cat(
                [
                    torch.ones(n_tracks, dtype=torch.float32),
                    torch.zeros(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
            "is_topo": torch.cat(
                [
                    torch.zeros(n_tracks, dtype=torch.float32),
                    torch.ones(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
        }

        node_raw_features = {
            "raw_e": torch.cat([torch.zeros(n_tracks, dtype=torch.float32), topo_e], -1),
            "raw_pt": torch.cat(
                [
                    track_pt,
                    torch.zeros(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
            "raw_eta": torch.cat([track_eta, topo_eta], -1),
            "raw_phi": torch.cat([track_phi, topo_phi], -1),
            "sinphi": torch.cat([track_sinphi, topo_sinphi], -1),
            "cosphi": torch.cat([track_cosphi, topo_cosphi], -1),
            "is_track": torch.cat(
                [
                    torch.ones(n_tracks, dtype=torch.float32),
                    torch.zeros(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
        }

        for key, val in node_features.items():
            val = do_padding(val, self.max_nodes)
            node_features[key] = val

        for key, val in node_raw_features.items():
            val = do_padding(val, self.max_nodes)
            node_raw_features[key] = val

        particle_class = self.particle_class[particle_start:particle_end]

        track_particle_idx = torch.tensor(self.aux_data_array["track_particle_idx"][idx], dtype=torch.int64)
        if self.is_inference:
            trackless_particle_mask = torch.zeros_like(particle_class, dtype=torch.bool)
        else:
            trackless_particle_mask = torch.ones_like(particle_class, dtype=torch.bool)
            trackless_particle_mask[track_particle_idx] = False

            trackless_chhad_and_e_mask = trackless_particle_mask & (particle_class < 2)
            trackless_muon_mask = trackless_particle_mask & (particle_class == 2)

            # trackless ch hads and es become nu hads and photons (+3)
            particle_class[trackless_chhad_and_e_mask] += 3

            # trackless muons become neutral hadrons
            particle_class[trackless_muon_mask] = 3

        is_charged = particle_class < 3

        particle_data = {
            "e": self.full_data_array["particle_e"][particle_start:particle_end],
            "pt": self.full_data_array["particle_pt"][particle_start:particle_end],
            "eta": self.full_data_array["particle_eta"][particle_start:particle_end],
            "sinphi": torch.sin(self.full_data_array["particle_phi"][particle_start:particle_end]),
            "cosphi": torch.cos(self.full_data_array["particle_phi"][particle_start:particle_end]),
            "class": particle_class,
            "is_charged": is_charged,
        }
        particle_data = self.scaler.transform(particle_data)
        for key, val in particle_data.items():
            val = do_padding(val, self.num_objects)
            particle_data[key] = val

        has_track = torch.zeros(self.num_objects, dtype=bool)
        has_track[:n_particles] = ~trackless_particle_mask

        node_q_mask = torch.zeros(self.max_nodes, dtype=bool)
        node_q_mask[:n_nodes] = True

        incidence_matrix = np.zeros((self.num_objects, n_nodes))
        indicator = torch.zeros(self.num_objects)

        part_idx = self.aux_data_array["track_particle_idx"][idx]
        track_idx = np.arange(len(part_idx))
        if self.is_inference:
            part_idx[part_idx < 0] = 0
        # print(incidence_matrix.shape, part_idx.shape, track_idx.shape, idx)
        incidence_matrix[part_idx, track_idx] = 1.0

        topo_idx = self.aux_data_array["topo2particle_topo_idx"][idx]
        part_idx = self.aux_data_array["topo2particle_particle_idx"][idx]
        incidence_matrix[part_idx, topo_idx + n_tracks] = self.aux_data_array["topo2particle_energy"][idx]

        # check for TC w/o associated particles
        if (incidence_matrix.sum(axis=0) == 0).any():
            noisy_cols = np.where(incidence_matrix.sum(axis=0) == 0)[0]
            fake_rows = np.arange(len(noisy_cols)) + n_particles
            if not (fake_rows < self.num_objects).all():
                print(f"Warning: fake_rows go beyond maximum ({self.num_objects}) particles. Dropping them!")
                noisy_cols = noisy_cols[fake_rows < self.num_objects]
                fake_rows = fake_rows[fake_rows < self.num_objects]
            incidence_matrix[fake_rows, noisy_cols] = 1.0

        # normalize
        incidence_matrix /= np.clip(incidence_matrix.sum(axis=0, keepdims=True), a_min=1e-6, a_max=None)

        incidence = torch.tensor(incidence_matrix, dtype=torch.float32)

        incidence = torch.nn.functional.pad(incidence, (0, self.max_nodes - n_nodes, 0, 0))
        # update the indicator
        is_not_res_mask = particle_class < 5
        indicator[:n_particles][is_not_res_mask] = 1.0

        node_inp_features = torch.stack(list(node_features.values()), dim=-1)

        return {
            "node_inp_features": node_inp_features,
            "node_raw_features": node_raw_features,
            "particle_data": particle_data,
            "incidence_truth": incidence,
            "indicator_truth": indicator,
            "node_q_mask": node_q_mask,
        }

    def __getitem__(self, idx):
        """Use .unsqueeze(0) to add in the dummy batch dimension (length 1 always)."""
        inputs = {}
        labels = {}

        # load event
        data_dict = self.load_event(idx)

        inputs = {
            "node_features": data_dict["node_inp_features"],
            "node_valid": data_dict["node_q_mask"],
            "node_e": data_dict["node_raw_features"]["raw_e"],
            "node_pt": data_dict["node_raw_features"]["raw_pt"],
            "node_eta": data_dict["node_raw_features"]["raw_eta"],
            "node_phi": data_dict["node_raw_features"]["raw_phi"],
            "node_sinphi": data_dict["node_raw_features"]["sinphi"],
            "node_cosphi": data_dict["node_raw_features"]["cosphi"],
            "node_is_track": data_dict["node_raw_features"]["is_track"],
        }

        # set class labels (5 is residual & fake, 0-4 are the real classes)
        class_labels = data_dict["particle_data"]["class"].long()
        class_labels[data_dict["indicator_truth"] == 0] = 5

        labels["particle_class"] = class_labels.long()
        labels["particle_valid"] = data_dict["indicator_truth"].bool()
        labels["node_valid"] = data_dict["node_q_mask"].bool()

        # get masks
        incidence_mask = data_dict["incidence_truth"] > self.incidence_cutval
        labels["particle_node_valid"] = incidence_mask

        # get incidence matrix label
        labels["particle_incidence"] = data_dict["incidence_truth"]

        # regression targets
        for label in self.targets["particle"]:
            tgt = torch.full((self.num_objects,), torch.nan)  # number of reconstructed tracks
            tgt[: self.n_particles[idx]] = data_dict["particle_data"][label][: self.n_particles[idx]]
            labels[f"particle_{label}"] = tgt

        labels["event_number"] = torch.tensor(self.event_number[idx], dtype=torch.int64)

        return inputs, labels

    def init_label_dicts(self):
        # charged hadron: 0, electron: 1, muon: 2, neutral hadron: 3, photon: 4, residual: 5, neutrino: -1
        self.class_labels = {
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
            311: 3,  # K0
            -311: 3,
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
            1000010048: 0,  # ?
            1000020032: 0,  # ?
            -999: 5,  # residual
            -12: -1,
            12: -1,  # nu_e
            -14: -1,
            14: -1,  # nu_mu
            -16: -1,
            16: -1,  # nu_tau
        }

    def init_variables_list(self):
        self.track_variables = [
            "track_pt",
            "track_eta",
            "track_phi",
            "track_d0",
            "track_z0",
            "track_eta_int",
            "track_phi_int",
            "track_chi2",
            "track_ndf",
            "track_radiusofinnermosthit",
            "track_tanlambda",
            "track_omega",
        ]

        self.topo_variables = [
            "topo_eta",
            "topo_phi",
            "topo_rho",
            "topo_e",
            "topo_sigma_eta",
            "topo_sigma_phi",
            "topo_sigma_rho",
            "topo_energy_ecal",
            "topo_energy_hcal",
            "topo_energy_other",
        ]

        self.particle_variables = [
            "particle_e",
            "particle_pt",
            "particle_eta",
            "particle_phi",
            "particle_pdgid",
        ]

        self.aux_vars = [
            "particle_track_idx",
            "track_particle_idx",
            "topo2particle_topo_idx",
            "topo2particle_particle_idx",
            "topo2particle_energy",
        ]


class PflowDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        valid_path: str,
        batch_size: int,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        scale_dict_path: str,
        test_path: str | None = None,
        pin_memory: bool = True,
        test_suff: str | None = None,
        **kwargs,
    ):
        super().__init__()

        self.train_path = train_path
        self.valid_path = valid_path
        self.batch_size = batch_size
        self.test_path = test_path
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.pin_memory = pin_memory
        self.test_suff = test_suff
        self.scale_dict_path = scale_dict_path
        self.kwargs = kwargs

    def setup(self, stage: str):
        if self.trainer.is_global_zero:
            print("-" * 100)

        # create training and validation datasets
        if stage == "fit":
            self.train_dset = CLICDataset(
                filepath=self.train_path,
                num_events=self.num_train,
                scale_dict_path=self.scale_dict_path,
                **self.kwargs,
            )

        if stage == "fit":
            self.val_dset = CLICDataset(
                filepath=self.valid_path,
                num_events=self.num_val,
                scale_dict_path=self.scale_dict_path,
                **self.kwargs,
            )

        # Only print train/val dataset details when actually training
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_path is not None, "No test file specified, see --data.test_path"
            self.test_dset = CLICDataset(
                filepath=self.test_path,
                num_events=self.num_test,
                scale_dict_path=self.scale_dict_path,
                **self.kwargs,
            )
            print(f"Created test dataset with {len(self.test_dset):,} events")

        if self.trainer.is_global_zero:
            print("-" * 100, "\n")

    def get_dataloader(self, stage: str, dataset: CLICDataset, shuffle: bool):
        print(f"Creating {stage} dataloader with {len(dataset):,} events")
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=None,
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        print("Instantiating train dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=True)

    def val_dataloader(self):
        print("Instantiating validation dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self):
        print("Instantiating test dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)
