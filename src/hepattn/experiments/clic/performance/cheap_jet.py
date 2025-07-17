import numpy as np


class CheapJet:
    def __init__(self, fastjet_obj, n_const=None, constituent_idxs=None):
        self.pt = fastjet_obj.pt()
        self.eta = fastjet_obj.eta()
        self.phi = fastjet_obj.phi_std()  # let's use phi_std as default
        self.phi_std = fastjet_obj.phi_std()
        self.mass = fastjet_obj.m()
        self.e = fastjet_obj.e()
        self.n_constituents = n_const
        self.constituent_idxs = constituent_idxs

    @classmethod
    def alternate(cls, pt, eta, phi, mass=None, e=None, n_const=None, constituent_idxs=None):
        instance = cls.__new__(cls)
        instance.pt = pt
        instance.eta = eta
        instance.phi = phi
        instance.phi_std = (phi + np.pi) % (2 * np.pi) - np.pi
        instance.mass = mass
        instance.e = e
        instance.n_constituents = n_const
        instance.constituent_idxs = constituent_idxs
        return instance

    def __add__(self, other):
        raise NotImplementedError("This is not implemented yet.")

    def delta_r(self, other):
        deta = self.eta - other.eta
        dphi = self.phi - other.phi
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        return np.sqrt(deta**2 + dphi**2)
