# coding: utf-8

from __future__ import annotations

import os
import functools
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, ClassVar

import numpy as np

from tautaunn.util import phi_mpi_to_pi, top_info, boson_info, match, calc_mass, calc_energy, calc_mt, hh

masses = [
    250, 260, 270, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650,
    700, 750, 800, 850, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000,
]

spins = [0, 2]

br_hh_bbtt = 0.073056256

channels = {
    "mutau": 0,
    "etau": 1,
    "tautau": 2,
}
klub_extra_columns = [
    # "DNNoutSM_kl_1",
]
# "years" in all structures above actually mean "era", so define "datacard year" as the actual year of an era
# for datacard purposes, as, for instance, eras "2016APV" and "2016" are both considered as datacard year "2016"
datacard_years = {
    "2016APV": "2016",
    "2016": "2016",
    "2017": "2017",
    "2018": "2018",
    "2022": "2022",
    "2022EE": "2022",
    "2023": "2023",
    "2023BPix": "2023",
    "2024": "2024"
}

processes = OrderedDict({
    "TT": {
        "id": 1,
        "sample_patterns": ["TT_*", "TTto*"],
    },
    "ST": {
        "id": 2,
        "sample_patterns": ["ST_*"],
    },
    "DY": {
        "id": 3,
        "sample_patterns": ["DY_*", "DYto*"],
    },
    "W": {
        "id": 4,
        "sample_patterns": ["WJets_*"],
    },
    "EWK": {
        "id": 5,
        "sample_patterns": ["EWK*"],
    },
    "WW": {
        "id": 6,
        "sample_patterns": ["WW"],
    },
    "WZ": {
        "id": 7,
        "sample_patterns": ["WZ"],
    },
    "ZZ": {
        "id": 8,
        "sample_patterns": ["ZZ"],
    },
    "VVV": {
        "id": 9,
        "sample_patterns": ["WWW", "WWZ", "WZZ", "ZZZ"],
    },
    "TTV": {
        "id": 10,
        "sample_patterns": ["TTWJets*", "TTZTo*"],
    },
    "TTVV": {
        "id": 11,
        "sample_patterns": ["TTWW", "TTWZ", "TTZZ"],
    },
    "ggH_htt": {
        "id": 12,
        "sample_patterns": ["GluGluHToTauTau"],
    },
    "qqH_htt": {
        "id": 13,
        "sample_patterns": ["VBFHToTauTau"],
    },
    "ZH_htt": {
        "id": 14,
        "sample_patterns": ["ZHToTauTau"],
    },
    "WH_htt": {
        "id": 15,
        "sample_patterns": ["WminusHToTauTau", "WplusHToTauTau"],
    },
    "ttH_hbb": {
        "id": 16,
        "sample_patterns": ["ttHTobb"],
    },
    "ttH_htt": {
        "id": 17,
        "sample_patterns": ["ttHToTauTau"],
    },
    "ggHH_hbbhtt": {
        "id": 18,
        "sample_patterns": ["ggHH*"],
    },
    "qqHH_hbbhtt": {
        "id": 19,
        "sample_patterns": ["qqHH*"],
    },
    "QCD": {
        "id": 20,
        "sample_patterns": [],
    },
    **{
        f"ggf_spin_{spin}_mass_{mass}_hbbhtt": {
            "id": 0,
            "sample_patterns": [f"{resonance}{mass}"],
            "spin": spin,
            "mass": mass,
            "signal": True,
        }
        for mass in masses
        for spin, resonance in zip(spins, ["Rad", "Grav"])
    },
    "data_mu": {
        "sample_patterns": ["Muon*"],
        "data": True,
        "channels": ["mutau"],
    },
    "data_egamma": {
        "sample_patterns": ["EGamma*"],
        "data": True,
        "channels": ["etau"],
    },
    "data_tau": {
        "sample_patterns": ["Tau*"],
        "data": True,
        "channels": ["mutau", "etau", "tautau"],
    },
    "data_met": {
        "sample_patterns": ["MET*"],
        "data": True,
        "channels": ["mutau", "etau", "tautau"],
    },
})


@dataclass
class ActivationSetting:

    # name of the activation as understood by tf.keras.layers.Activation
    name: str
    # name of the kernel initializer as understood by tf.keras.layers.Dense
    weight_init: str
    # whether to apply batch normalization before or after the activation (and if at all)
    batch_norm: tuple[bool, bool]
    # name of the dropout layer under tf.keras.layers
    dropout_name: str = "Dropout"


activation_settings = {
    "elu": ActivationSetting("ELU", "he_uniform", (True, False)),
    "relu": ActivationSetting("ReLU", "he_uniform", (False, True)),
    "prelu": ActivationSetting("PReLU", "he_normal", (True, False)),
    "selu": ActivationSetting("selu", "lecun_normal", (False, False), "AlphaDropout"),
    "tanh": ActivationSetting("tanh", "glorot_normal", (True, False)),
    "softmax": ActivationSetting("softmax", "glorot_normal", (True, False)),
    "swish": ActivationSetting("swish", "glorot_uniform", (True, False)),
}

skim_dirs = {
    "2016APV": os.environ["TN_SKIMS_2016APV"],
    "2016": os.environ["TN_SKIMS_2016"],
    "2017": os.environ["TN_SKIMS_2017"],
    "2018": os.environ["TN_SKIMS_2018"],
    "2022": os.environ["TN_SKIMS_2022"],
    "2022EE": os.environ["TN_SKIMS_2022EE"],
    "2023": os.environ["TN_SKIMS_2023"],
    "2023BPix": os.environ["TN_SKIMS_2023BPix"],
    "2024": os.environ["TN_SKIMS_2024"]
}


@functools.cache
def get_all_skim_names() -> dict[str, list[str]]:
    # note: VBF signals are skipped!
    return {
        year: [
            d for d in os.listdir(skim_dir)
            if (
                os.path.isdir(os.path.join(skim_dir, d))
            )
        ]
        for year, skim_dir in skim_dirs.items()
    }


@functools.cache
def get_skim_names(skim_dir: str) -> dict[str, list[str]]:
    # note: VBF signals are skipped!
    skim_dir = os.path.expandvars(os.path.expanduser(skim_dir))
    return [
        d for d in os.listdir(skim_dir)
        if (
            os.path.isdir(os.path.join(skim_dir, d))
        )
    ]


luminosities = {
    "2016APV": 19_500.0,
    "2016": 16_800.0,
    "2017": 41_480.0,
    "2018": 59_830.0,
    "2022": 7_980.0,
    "2022EE": 23_589.0,
    "2023": 18_063.0,
    "2023BPix": 9_693.0,
    "2024": 107_900.0
}

btag_wps = {
    "2016APV": {
        "loose": 0.0508,
        "medium": 0.2598,
    },
    "2016": {
        "loose": 0.0480,
        "medium": 0.2489,
    },
    "2017": {
        "loose": 0.0532,
        "medium": 0.3040,
    },
    "2018": {
        "loose": 0.0490,
        "medium": 0.2783,
    },
}

pnet_wps = {
    "2016APV": 0.9088,
    "2016": 0.9137,
    "2017": 0.9105,
    "2018": 0.9172,
}


category_indices = {
    "mutau_res1b": 1,
    "mutau_res2b": 2,
    "mutau_boosted": 3,
    "etau_res1b": 4,
    "etau_res2b": 5,
    "etau_boosted": 6,
    "tautau_res1b": 7,
    "tautau_res2b": 8,
    "tautau_boosted": 9,
}


@dataclass
class Sample:
    """
    Example:
    - name: "ggF_Radion_m350"
    - skim_name: "2016APV_ggF_Radion_m350"
    - directory_name: same as name, only existing for backwards compatibility
    - year: "2016APV"  (this is more like a "campaign")
    - year_int: 2016
    - year_flag: 0
    - spin: 0
    - mass: 350.0
    """

    name: str
    year: str
    label: int | None = None
    loss_weight: float = 1.0
    spin: int = -1
    mass: float = -1.0
    category: str = ''
    version: str = ''

    YEAR_FLAGS: ClassVar[dict[str, int]] = {
        "2016APV": 0,
        "2016": 1,
        "2017": 2,
        "2018": 3,
        "2022": 4,
        "2022EE": 5,
        "2023": 6,
        "2023BPix": 7,
        "2024": 8
    }

    def __hash__(self) -> int:
        return hash(self.hash_values)

    @property
    def hash_values(self) -> tuple[Any]:
        return (self.skim_name, self.year, self.label, self.loss_weight, self.spin, self.mass)

    @property
    def skim_name(self) -> str:
        return f"{self.year}_{self.name}"

    @property
    def directory_name(self) -> str:
        return self.name

    @property
    def year_int(self) -> int:
        return int(self.year[:4])

    @property
    def year_flag(self) -> int:
        return self.YEAR_FLAGS[self.year]

    @property
    def is_signal(self) -> bool:
        return self.name.startswith(("Rad", "Grav"))

    @property
    def is_data(self) -> bool:
        return self.name.startswith(("Tau", "Muon", "EGamma", "MET"))

    def with_label_and_loss_weight(self, label: int | None, loss_weight: float = 1.0) -> Sample:
        return self.__class__(
            name=self.name,
            year=self.year,
            label=label,
            loss_weight=loss_weight,
            spin=self.spin,
            mass=self.mass,
            category=self.category,
            version=self.version
        )


all_samples = [
    *[
        Sample(f"{res_name}{mass}", year=year, spin=spin, mass=float(mass))
        for year in luminosities.keys()
        for spin, res_name in [(0, "Rad"), (2, "Grav")]
        for mass in masses
    ],
    *[
        Sample(f"TT_{tt_channel}Lep", year=year)
        for year in luminosities.keys()
        for tt_channel in ["Fully", "Semi"]
    ],
    *[
        Sample(f"DY_{dy_suffix}", year=year)
        for year in luminosities.keys()
        for dy_suffix in [
            "Incl",
            "0J", "1J", "2J",
            "PtZ0To50", "PtZ100To250", "PtZ250To400", "PtZ400To650", "PtZ50To100", "PtZ650ToInf",
        ]
    ],
    *[
        Sample("ttHToTauTau", year=year)
        for year in luminosities.keys()
    ],
]


# helper to get a single sample by name and year
def get_sample(skim_name: str, silent: bool = False) -> Sample | None:
    for sample in all_samples:
        if sample.skim_name == skim_name:
            return sample
    if silent:
        return None
    raise ValueError(f"sample with skim_name {skim_name} not found")


# helper to select samples with skim_name patterns
def select_samples(*patterns):
    samples = []
    for pattern in patterns:
        for sample in all_samples:
            if match(sample.skim_name, pattern) and sample not in samples:
                samples.append(sample)
    return samples


train_masses_central = "320|350|400|450|500|550|600|650|700|750|800|850|900|1000|1250|1500|1750"
train_masses_all = "250|260|270|280|300|320|350|400|450|500|550|600|650|700|750|800|850|900|1000|1250|1500|1750|2000|2500|3000"  # noqa
sample_sets = {
    "default_2016APV": (samples_default_2016APV := select_samples(
        rf"^2016APV_Rad({train_masses_all})$",
        rf"^2016APV_Grav({train_masses_all})$",
        r"^2016APV_DY_PtZ.*$",
        r"^2016APV_TT_(Fully|Semi)Lep$",
        r"^2016APV_ttHToTauTau*$",
    )),
    # TODO: adjust to actual 2016
    "default_2016": (samples_default_2016 := select_samples(
        rf"^2016_Rad({train_masses_all})$",
        rf"^2016_Grav({train_masses_all})$",
        r"^2016_DY_PtZ.*$",
        r"^2016_TT_(Fully|Semi)Lep$",
        r"^2016_ttHToTauTau*$",
    )),
    "default_2016all": samples_default_2016APV + samples_default_2016,
    "default_2017": (samples_default_2017 := select_samples(
        rf"^2017_Rad({train_masses_all})$",
        rf"^2017_Grav({train_masses_all})$",
        r"^2017_DY_PtZ.*$",
        r"^2017_TT_(Fully|Semi)Lep$",
        r"^2017_ttHToTauTau*$",
    )),
    "default_2018": (samples_default_2018 := select_samples(
        rf"^2018_Rad({train_masses_all})$",
        rf"^2018_Grav({train_masses_all})$",
        r"^2018_DY_PtZ.*$",
        r"^2018_TT_(Fully|Semi)Lep$",
        r"^2018_ttHToTauTau*$",
    )),
    "default_1617": samples_default_2016APV + samples_default_2016 + samples_default_2017,
    "default": samples_default_2016APV + samples_default_2016 + samples_default_2017 + samples_default_2018,
    "test": select_samples(
        "2017_ggF_BulkGraviton_m500",
        "2017_ggF_BulkGraviton_m550",
        "2017_DY_amc_PtZ_0To50",
        "2017_DY_amc_PtZ_100To250",
        "2017_TT_semiLep",
    ),
    "vbf": [
        Sample("qqHH_CV_1_C2V_1_kl_1_hbbhtt", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_1_C2V_0_kl_1_hbbhtt", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_1p74_C2V_1p37_kl_14p4_hbbhtt", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m0p012_C2V_0p030_kl_10p2_hbbhtt", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m0p758_C2V_1p44_kl_m19p3_hbbhtt", year="2022EE", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m0p962_C2V_0p959_kl_m1p43_hbbhtt", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m1p21_C2V_1p94_kl_m0p94_hbbhtt", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m1p60_C2V_2p72_kl_m1p36_hbbhtt", year="2022EE", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m1p83_C2V_3p57_kl_m3p39_hbbhtt", year="2022EE", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_2p12_C2V_3p87_kl_m5p96_hbbhtt", year="2022EE", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=0, spin=0, mass=250.0),
        # Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2022EE", category="vbf_loose", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("DYto2[E,M]*", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_0J", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_1J", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_2J", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("TTto2L2Nu", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=1),
        Sample("TTtoLNu2Q", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=1),
        Sample("TTto4Q", year="2022EE", category="vbf_resolved", version="Prod_26_01", label=1),
        Sample("qqHH_CV_1_C2V_1_kl_1_hbbhtt", year="2022", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_1_C2V_0_kl_1_hbbhtt", year="2022", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_1p74_C2V_1p37_kl_14p4_hbbhtt", year="2022", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m0p012_C2V_0p030_kl_10p2_hbbhtt", year="2022", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m0p758_C2V_1p44_kl_m19p3_hbbhtt", year="2022", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m0p962_C2V_0p959_kl_m1p43_hbbhtt", year="2022", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m1p21_C2V_1p94_kl_m0p94_hbbhtt", year="2022", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m1p60_C2V_2p72_kl_m1p36_hbbhtt", year="2022", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m1p83_C2V_3p57_kl_m3p39_hbbhtt", year="2022", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_2p12_C2V_3p87_kl_m5p96_hbbhtt", year="2022", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2022", category="vbf_resolved", version="Prod_26_01", label=0, spin=0, mass=250.0),
        # Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2022", category="vbf_loose", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2022", category="vbf_resolved", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2022", category="vbf_resolved", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("DYto2[E,M]*", year="2022", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_0J", year="2022", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_1J", year="2022", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_2J", year="2022", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("TTto2L2Nu", year="2022", category="vbf_resolved", version="Prod_26_01", label=1),
        Sample("TTtoLNu2Q", year="2022", category="vbf_resolved", version="Prod_26_01", label=1),
        Sample("TTto4Q", year="2022", category="vbf_resolved", version="Prod_26_01", label=1),
        Sample("qqHH_CV_1_C2V_1_kl_1_hbbhtt", year="2023", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_1_C2V_0_kl_1_hbbhtt", year="2023", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_1p74_C2V_1p37_kl_14p4_hbbhtt", year="2023", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m0p012_C2V_0p030_kl_10p2_hbbhtt", year="2023", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m0p758_C2V_1p44_kl_m19p3_hbbhtt", year="2023", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m0p962_C2V_0p959_kl_m1p43_hbbhtt", year="2023", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m1p21_C2V_1p94_kl_m0p94_hbbhtt", year="2023", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m1p60_C2V_2p72_kl_m1p36_hbbhtt", year="2023", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m1p83_C2V_3p57_kl_m3p39_hbbhtt", year="2023", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_2p12_C2V_3p87_kl_m5p96_hbbhtt", year="2023", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2023", category="vbf_resolved", version="Prod_26_01", label=0, spin=0, mass=250.0),
        # Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2023", category="vbf_resolved", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2023", category="vbf_resolved", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2023", category="vbf_resolved", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("DYto2[E,M]*", year="2023", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_0J", year="2023", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_1J", year="2023", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_2J", year="2023", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("TTto2L2Nu", year="2023", category="vbf_resolved", version="Prod_26_01", label=1),
        Sample("TTtoLNu2Q", year="2023", category="vbf_resolved", version="Prod_26_01", label=1),
        Sample("TTto4Q", year="2023", category="vbf_resolved", version="Prod_26_01", label=1),
        Sample("qqHH_CV_1_C2V_1_kl_1_hbbhtt", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_1_C2V_0_kl_1_hbbhtt", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_1p74_C2V_1p37_kl_14p4_hbbhtt", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m0p012_C2V_0p030_kl_10p2_hbbhtt", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m0p758_C2V_1p44_kl_m19p3_hbbhtt", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m0p962_C2V_0p959_kl_m1p43_hbbhtt", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m1p21_C2V_1p94_kl_m0p94_hbbhtt", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m1p60_C2V_2p72_kl_m1p36_hbbhtt", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m1p83_C2V_3p57_kl_m3p39_hbbhtt", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_2p12_C2V_3p87_kl_m5p96_hbbhtt", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=0, spin=0, mass=250.0),
        # Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("DYto2[E,M]*", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_0J", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_1J", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_2J", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=2),
        Sample("TTto2L2Nu", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=1),
        Sample("TTtoLNu2Q", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=1),
        Sample("TTto4Q", year="2023BPix", category="vbf_resolved", version="Prod_26_01", label=1),
    ],
    "vbf_2024": [
         Sample("qqHH_CV_1_C2V_1_kl_1_hbbhtt", year="2024", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_1_C2V_0_kl_1_hbbhtt", year="2024", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_1p74_C2V_1p37_kl_14p4_hbbhtt", year="2024", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m0p012_C2V_0p030_kl_10p2_hbbhtt", year="2024", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m0p758_C2V_1p44_kl_m19p3_hbbhtt", year="2024", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m0p962_C2V_0p959_kl_m1p43_hbbhtt", year="2024", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("qqHH_CV_m1p21_C2V_1p94_kl_m0p94_hbbhtt", year="2024", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m1p60_C2V_2p72_kl_m1p36_hbbhtt", year="2024", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_m1p83_C2V_3p57_kl_m3p39_hbbhtt", year="2024", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        # Sample("qqHH_CV_2p12_C2V_3p87_kl_m5p96_hbbhtt", year="2024", category="vbf_loose", version="Prod_26_01", label=3, spin=0, mass=250.0),
        Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2024", category="vbf_loose", version="Prod_26_01", label=0, spin=0, mass=250.0),
        # Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2024", category="vbf_loose", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2024", category="vbf_loose", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2024", category="vbf_loose", version="Prod_26_01", label=0, spin=0, mass=250.0),
        Sample("DYto2[E,M]*", year="2024", category="vbf_loose", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_0J", year="2024", category="vbf_loose", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_1J", year="2024", category="vbf_loose", version="Prod_26_01", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_2J", year="2024", category="vbf_loose", version="Prod_26_01", label=2),
        Sample("TTto2L2Nu", year="2024", category="vbf_loose", version="Prod_26_01", label=1),
        Sample("TTtoLNu2Q", year="2024", category="vbf_loose", version="Prod_26_01", label=1),
        Sample("TTto4Q", year="2024", category="vbf_loose", version="Prod_26_01", label=1),
    ],
    "ggf": [
        #res1-2b
        Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2022EE", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2022EE", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2022EE", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2022EE", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("DYto2[E,M]*", year="2022EE", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_0J", year="2022EE", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_1J", year="2022EE", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_2J", year="2022EE", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("TTto2L2Nu", year="2022EE", category="res[1,2]b", version="Prod_26_03", label=1),
        Sample("TTtoLNu2Q", year="2022EE", category="res[1,2]b", version="Prod_26_03", label=1),
        Sample("TTto4Q", year="2022EE", category="res[1,2]b", version="Prod_26_03", label=1),
        Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2022", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2022", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2022", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2022", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("DYto2[E,M]*", year="2022", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_0J", year="2022", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_1J", year="2022", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_2J", year="2022", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("TTto2L2Nu", year="2022", category="res[1,2]b", version="Prod_26_03", label=1),
        Sample("TTtoLNu2Q", year="2022", category="res[1,2]b", version="Prod_26_03", label=1),
        Sample("TTto4Q", year="2022", category="res[1,2]b", version="Prod_26_03", label=1),
        Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2023", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2023", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2023", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2023", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("DYto2[E,M]*", year="2023", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_0J", year="2023", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_1J", year="2023", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_2J", year="2023", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("TTto2L2Nu", year="2023", category="res[1,2]b", version="Prod_26_03", label=1),
        Sample("TTtoLNu2Q", year="2023", category="res[1,2]b", version="Prod_26_03", label=1),
        Sample("TTto4Q", year="2023", category="res[1,2]b", version="Prod_26_03", label=1),
        Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2023BPix", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2023BPix", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2023BPix", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2023BPix", category="res[1,2]b", version="Prod_26_03", label=0, spin=0, mass=250.0),
        Sample("DYto2[E,M]*", year="2023BPix", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_0J", year="2023BPix", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_1J", year="2023BPix", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_2J", year="2023BPix", category="res[1,2]b", version="Prod_26_03", label=2),
        Sample("TTto2L2Nu", year="2023BPix", category="res[1,2]b", version="Prod_26_03", label=1),
        Sample("TTtoLNu2Q", year="2023BPix", category="res[1,2]b", version="Prod_26_03", label=1),
        Sample("TTto4Q", year="2023BPix", category="res[1,2]b", version="Prod_26_03", label=1),
        # boosted
    #     Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2022EE", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2022EE", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2022EE", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2022EE", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("DYto2[E,M]*", year="2022EE", category="boosted", version="Prod_26_03", label=2),
    #     #Sample("DYto2Tau-2Jets_MLL-50_0J", year="2022EE", category="boosted", version="Prod_26_03", label=2),
    #     Sample("DYto2Tau-2Jets_MLL-50_1J", year="2022EE", category="boosted", version="Prod_26_03", label=2),
    #     Sample("DYto2Tau-2Jets_MLL-50_2J", year="2022EE", category="boosted", version="Prod_26_03", label=2),
    #     Sample("TTto2L2Nu", year="2022EE", category="boosted", version="Prod_26_03", label=1),
    #     Sample("TTtoLNu2Q", year="2022EE", category="boosted", version="Prod_26_03", label=1),
    #     Sample("TTto4Q", year="2022EE", category="boosted", version="Prod_26_03", label=1),
    #     Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2022", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2022", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2022", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2022", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("DYto2[E,M]*", year="2022", category="boosted", version="Prod_26_03", label=2),
    #     #Sample("DYto2Tau-2Jets_MLL-50_0J", year="2022", category="boosted", version="Prod_26_03", label=2),
    #     Sample("DYto2Tau-2Jets_MLL-50_1J", year="2022", category="boosted", version="Prod_26_03", label=2),
    #     Sample("DYto2Tau-2Jets_MLL-50_2J", year="2022", category="boosted", version="Prod_26_03", label=2),
    #     Sample("TTto2L2Nu", year="2022", category="boosted", version="Prod_26_03", label=1),
    #     Sample("TTtoLNu2Q", year="2022", category="boosted", version="Prod_26_03", label=1),
    #     Sample("TTto4Q", year="2022", category="boosted", version="Prod_26_03", label=1),
    #     Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2023", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2023", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2023", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2023", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("DYto2[E,M]*", year="2023", category="boosted", version="Prod_26_03", label=2),
    #     #Sample("DYto2Tau-2Jets_MLL-50_0J", year="2023", category="boosted", version="Prod_26_03", label=2),
    #     Sample("DYto2Tau-2Jets_MLL-50_1J", year="2023", category="boosted", version="Prod_26_03", label=2),
    #     Sample("DYto2Tau-2Jets_MLL-50_2J", year="2023", category="boosted", version="Prod_26_03", label=2),
    #     Sample("TTto2L2Nu", year="2023", category="boosted", version="Prod_26_03", label=1),
    #     Sample("TTtoLNu2Q", year="2023", category="boosted", version="Prod_26_03", label=1),
    #     Sample("TTto4Q", year="2023", category="boosted", version="Prod_26_03", label=1),
    #     Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2023BPix", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2023BPix", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2023BPix", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2023BPix", category="boosted", version="Prod_26_03", label=0, spin=0, mass=250.0),
    #     Sample("DYto2[E,M]*", year="2023BPix", category="boosted", version="Prod_26_03", label=2),
    #     #Sample("DYto2Tau-2Jets_MLL-50_0J", year="2023BPix", category="boosted", version="Prod_26_03", label=2),
    #     Sample("DYto2Tau-2Jets_MLL-50_1J", year="2023BPix", category="boosted", version="Prod_26_03", label=2),
    #     Sample("DYto2Tau-2Jets_MLL-50_2J", year="2023BPix", category="boosted", version="Prod_26_03", label=2),
    #     Sample("TTto2L2Nu", year="2023BPix", category="boosted", version="Prod_26_03", label=1),
    #     Sample("TTtoLNu2Q", year="2023BPix", category="boosted", version="Prod_26_03", label=1),
    #     Sample("TTto4Q", year="2023BPix", category="boosted", version="Prod_26_03", label=1),
    ],
    "ggf_2024": [
        #res1-2b
        Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2024", category="res[1,2]b", version="Prod_26_04", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2024", category="res[1,2]b", version="Prod_26_04", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2024", category="res[1,2]b", version="Prod_26_04", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2024", category="res[1,2]b", version="Prod_26_04", label=0, spin=0, mass=250.0),
        Sample("DYto2[E,M]*", year="2024", category="res[1,2]b", version="Prod_26_04", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_0J", year="2024", category="res[1,2]b", version="Prod_26_04", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_1J", year="2024", category="res[1,2]b", version="Prod_26_04", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_2J", year="2024", category="res[1,2]b", version="Prod_26_04", label=2),
        Sample("TTto2L2Nu", year="2024", category="res[1,2]b", version="Prod_26_04", label=1),
        Sample("TTtoLNu2Q", year="2024", category="res[1,2]b", version="Prod_26_04", label=1),
        Sample("TTto4Q", year="2024", category="res[1,2]b", version="Prod_26_04", label=1),
        #boosted
        Sample("ggHH_kl_1_kt_1_c2_0_hbbhtt", year="2024", category="boosted", version="Prod_26_04", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_0_kt_1_c2_0_hbbhtt", year="2024", category="boosted", version="Prod_26_04", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_2p45_kt_1_c2_0_hbbhtt", year="2024", category="boosted", version="Prod_26_04", label=0, spin=0, mass=250.0),
        Sample("ggHH_kl_5_kt_1_c2_0_hbbhtt", year="2024", category="boosted", version="Prod_26_04", label=0, spin=0, mass=250.0),
        Sample("DYto2[E,M]*", year="2024", category="boosted", version="Prod_26_04", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_0J", year="2024", category="boosted", version="Prod_26_04", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_1J", year="2024", category="boosted", version="Prod_26_04", label=2),
        Sample("DYto2Tau-2Jets_MLL-50_2J", year="2024", category="boosted", version="Prod_26_04", label=2),
        Sample("TTto2L2Nu", year="2024", category="boosted", version="Prod_26_04", label=1),
        Sample("TTtoLNu2Q", year="2024", category="boosted", version="Prod_26_04", label=1),
        Sample("TTto4Q", year="2024", category="boosted", version="Prod_26_04", label=1),
    ]
}

# label information
# (sample patterns are evaluated on top of those selected by sample_sets)
label_sets = {
    "binary": {
        0: {"name": "Signal", "sample_patterns": [r"^201\d.*_(Rad|Grav)\d+$"]},
        1: {"name": "Background", "sample_patterns": ["201*_DY*", "201*_TT*"]},
    },
    "multi3": {
        0: {"name": "HH", "sample_patterns": [r"^201\d.*_(Rad|Grav)\d+$", "ggHH*", "qqHH*"]},
        1: {"name": "TT", "sample_patterns": ["201*_TT*", "TTto*"]},
        2: {"name": "DY", "sample_patterns": ["201*_DY*", "DYto*"]},
    },
    "multi4": {
        0: {"name": "HH", "sample_patterns": [r"^201\d.*_(Rad|Grav)\d+$"]},
        1: {"name": "DY", "sample_patterns": ["201*_DY*"]},
        2: {"name": "TT", "sample_patterns": ["201*_TT*"]},
        3: {"name": "TTH", "sample_patterns": ["201*_ttHToTauTau*"]},
    },
    "quad": {
        0: {"name": "HH", "sample_patterns": [r"^201\d.*_(Rad|Grav)\d+$", "ggHH*"]},
        1: {"name": "TT", "sample_patterns": ["201*_TT*", "TTto*"]},
        2: {"name": "DY", "sample_patterns": ["201*_DY*", "DYto*"]},
        3: {"name": "VBF", "sample_patterns": ["qqHH*"]}
    },
    "HHvsVBF": {
        0: {"name": "HH", "sample_patterns": [r"^201\d.*_(Rad|Grav)\d+$", "ggHH*"]},
        1: {"name": "VBF", "sample_patterns": ["qqHH*"]}
    },
    "SigvsBkg": {
        0: {"name": "HH", "sample_patterns": [r"^201\d.*_(Rad|Grav)\d+$", "ggHH*"]},
        1: {"name": "TT", "sample_patterns": ["201*_TT*", "TTto*", "DYto*", "Wto*"]}
    },
}


def with_features(original, *, add=None, remove=None):
    features = list(original)
    if remove:
        remove = remove if isinstance(remove, list) else [remove]
        features = [f for f in features if not any(match(f, p) for p in remove)]
    if add:
        features += add if isinstance(add, list) else [add]
    return features


cont_feature_sets = {
    "reg": (cont_features_reg := [
        "met_px", "met_py", "dmet_resp_px", "dmet_resp_py", "dmet_reso_px",
        "met_cov00", "met_cov01", "met_cov11",
        "ditau_deltaphi", "ditau_deltaeta",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "dxy", "dz", "iso"]
        ],
        *[
            f"bjet{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e", "btag_deepFlavor", "cID_deepFlavor",
                "pnet_bb", "pnet_cc", "pnet_b", "pnet_c", "pnet_g", "pnet_uds", "pnet_pu", "pnet_undef",
                "HHbtag",
            ]
        ],
    ]),
    "reg2": (cont_features_reg2 := [
        # order is important here since it is used as is for the tauNN
        "met_px", "met_py", "dmet_resp_px", "dmet_resp_py", "dmet_reso_px",
        "ditau_deltaphi", "ditau_deltaeta",
        "dau1_px", "dau1_py", "dau1_pz", "dau1_e", "dau1_iso",
        "dau2_px", "dau2_py", "dau2_pz", "dau2_e", "dau2_iso",
        "met_cov00", "met_cov01", "met_cov11",
        "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e", "bjet1_btag_deepFlavor", "bjet1_cID_deepFlavor",
        "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e", "bjet2_btag_deepFlavor", "bjet2_cID_deepFlavor",
    ]),
    "reg_nopnet": with_features(cont_features_reg, remove=["bjet*_pnet_*"]),
    "reg_nohhbtag": with_features(cont_features_reg, remove=["bjet*_HHbtag"]),
    "reg_nodf": with_features(cont_features_reg, remove=["bjet*_deepFlavor"]),
    "reg_nohl": with_features(cont_features_reg, remove=["ditau_*"]),
    "reg_nodau2iso": with_features(cont_features_reg, remove=["dau2_iso"]),
    "reg_nodaudxyz": with_features(cont_features_reg, remove=["dau*_dxy", "dau*_dz"]),
    "reg_nohhbtag_nohl": with_features(cont_features_reg, remove=["bjet*_HHbtag", "ditau_*"]),
    "post_meeting161": with_features(cont_features_reg, remove=["ditau_*", "dau*_iso", "dmet_*"]),
    "reg_reduced": (cont_features_reg_reduced := [
        "met_et", "met_cov00", "met_cov01", "met_cov11",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "dxy", "dz"]
        ],
        *[
            f"bjet{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "btag_deepFlavor", "cID_deepFlavor", "HHbtag"]
        ],
    ]),
    "reg_reduced_cid": (cont_features_reg_reduced_cid := with_features(
        cont_features_reg_reduced,
        remove=["bjet*_cID_deepFlavor"],
        add=["bjet1_PNetCvL", "bjet1_PNetCvB", "bjet2_PNetCvL", "bjet2_PNetCvB"],
    )),
    "reg_reduced_cid_pnet": [
        "met_et", "met_cov00", "met_cov01", "met_cov11",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "dxy", "dz"]
        ],
        *[
            f"bjet{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e",
                "btag_deepFlavor", "CvsB", "CvsL",
                "pnet_bb", "pnet_cc", "pnet_b", "pnet_c", "pnet_g", "pnet_uds", "pnet_pu", "pnet_undef",
                "HHbtag",
            ]
        ],
    ],
    "default_metrot": [
        "met_et", "met_cov00", "met_cov01", "met_cov11",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "dxy", "dz"]
        ],
        *[
            f"bjet{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e",
                "btag_deepFlavor", "cID_deepFlavor", "CvsB", "CvsL",
                "pnet_bb", "pnet_cc", "pnet_b", "pnet_c", "pnet_g", "pnet_uds", "pnet_pu", "pnet_undef",
                "HHbtag",
            ]
        ],
    ],
    "default_daurot": [
        "met_px", "met_py", "dmet_resp_px", "dmet_resp_py", "dmet_reso_px", "dmet_reso_py",
        "met_cov00", "met_cov01", "met_cov11",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "dxy", "dz"]
        ],
        *[
            f"bjet{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e",
                "btag_DeepFlavB", "cID_deepFlavor", "PNetCvB", "PNetCvL",
                "HHbtag",
            ]
        ],
    ],
    "default_daurot_fatjet": (cont_features_daurot_fatjet := [
        "met_px", "met_py",
        "met_cov00", "met_cov01", "met_cov11",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e"]
        ],
        *[
            f"bjet{i}_masked_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e",
                "btagDeepFlavB", "PNetCvB", "PNetCvL",
                "HHbtag",
            ]
        ],
        *[
            f"fatjet_masked_{feat}"
            #f"fatjet_{feat}"
            for feat in [
                 "e", "px", "py", "pz",
            ]
        ],
    ]),
    "default_daurot_fatjet_old": cont_features_daurot_fatjet + [
        "dau1_dxy", "dau1_dz", "dau2_dxy", "dau2_dz",
        "bjet1_masked_cID_deepFlavor", "bjet2_masked_cID_deepFlavor",
    ],
    "default_daurot_composite": cont_features_daurot_fatjet + [
        "htt_e", "htt_px", "htt_py", "htt_pz",
        "hbb_masked_e", "hbb_masked_px", "hbb_masked_py", "hbb_masked_pz",
        "htthbb_masked_e", "htthbb_masked_px", "htthbb_masked_py", "htthbb_masked_pz",
        "httfatjet_masked_e", "httfatjet_masked_px", "httfatjet_masked_py", "httfatjet_masked_pz",
    ],
    "default_daurot_composite_old": cont_features_daurot_fatjet + [
        "htt_e", "htt_px", "htt_py", "htt_pz",
        "hbb_masked_e", "hbb_masked_px", "hbb_masked_py", "hbb_masked_pz",
        "htthbb_masked_e", "htthbb_masked_px", "htthbb_masked_py", "htthbb_masked_pz",
        "httfatjet_masked_e", "httfatjet_masked_px", "httfatjet_masked_py", "httfatjet_masked_pz",
        "dau1_dxy", "dau1_dz", "dau2_dxy", "dau2_dz",
        "bjet1_masked_cID_deepFlavor", "bjet2_masked_cID_deepFlavor",
    ],
    "default_daurot_masked": [
        "met_px", "met_py", "dmet_resp_px", "dmet_resp_py", "dmet_reso_px", "dmet_reso_py",
        "met_cov00", "met_cov01", "met_cov11",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "dxy", "dz"]
        ],
        *[
            f"bjet{i}_masked_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e",
                "btag_deepFlavor", "cID_deepFlavor", "CvsB", "CvsL",
                "HHbtag",
            ]
        ],
    ],
    "full": (cont_features_full := cont_features_reg + [
        "tauH_e", "tauH_px", "tauH_py", "tauH_pz",
        "bH_e", "bH_px", "bH_py", "bH_pz",
        "HH_e", "HH_px", "HH_py", "HH_pz",
        "HHKin_mass",
        "top1_mass", "top2_mass", "W_distance", "Z_distance", "H_distance",
    ]),
    "full_svfit": cont_features_full + ["tauH_SVFIT_mass", "tauH_SVFIT_pt"],
    "reg_svfit": cont_features_reg + ["tauH_SVFIT_mass", "tauH_SVFIT_pt"],
    "class": [
        "bjet1_bID_deepFlavor", "bjet1_cID_deepFlavor", "bjet1_HHbtag",
        "bjet2_bID_deepFlavor", "bjet2_cID_deepFlavor", "bjet2_HHbtag",
        "dibjet_deltaR",
        "dau1_mt",
        "dau2_pt",
        "ditau_mt", "ditau_deltaR", "ditau_deltaeta",
        "tauH_SVFIT_e", "tauH_SVFIT_mass",
        "met_et",
        "top1_mass",
        "h_bb_mass",
        "hh_pt",
        "HHKin_mass_raw_chi2",
        "dphi_hbb_met",
        "deta_hbb_httvis",
        "HHKin_mass_raw",
        "diH_mass_met",
    ],
    "reg_v2": [
        "met_et",
        "met_cov00", "met_cov01", "met_cov11",
        "dau1_px", "dau1_py", "dau1_pz", "dau1_e", "dau1_dxy", "dau1_dz",
        "dau2_px", "dau2_py", "dau2_pz", "dau2_e", "dau2_dxy", "dau2_dz",
        "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e", "bjet1_btag_deepFlavor", "bjet1_cID_deepFlavor", "bjet1_HHbtag",
        "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e", "bjet2_btag_deepFlavor", "bjet2_cID_deepFlavor", "bjet2_HHbtag",
    ],
    "vbf": [
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e"]
        ],
        *[
            f"bjet{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e", "pnet_b", "pnet_CvsB", "pnet_CvsL", "HHbtag"
            ]
        ],
        *[
            f"nu{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz"
            ]
        ],
        "fatjet_px", "fatjet_py", "fatjet_pz", "fatjet_e",
        "htt_regr_px", "htt_regr_py", "htt_regr_pz", "htt_regr_e",
        "hbb_px", "hbb_py", "hbb_pz", "hbb_e",
        "httfatjet_regr_px", "httfatjet_regr_py", "httfatjet_regr_pz", "httfatjet_regr_e", 
        *[
            f"vbfjet{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e", "pnet_QvsG"
            ]
        ],
        "M_chi",
        "VBFjj_mass", "VBFdeltaR",
        "etaprod_bb", "etaprod_vbfjvbfj",
        "fwMoment_s_0", "fwMoment_1_0",
        "fwMoment_T_3"
    ],
    "vbf_b": [
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e"]
        ],
        *[
            f"bjet{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e", "pnet_b", "pnet_CvsB", "pnet_CvsL", "HHbtag"
            ]
        ],
        *[
            f"nu{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz"
            ]
        ],
        "fatjet_px", "fatjet_py", "fatjet_pz", "fatjet_e",
        "htt_regr_px", "htt_regr_py", "htt_regr_pz", "htt_regr_e",
        "hbb_px", "hbb_py", "hbb_pz", "hbb_e",
        "htthbb_regr_e", "htthbb_regr_px", "htthbb_regr_py", "htthbb_regr_pz",
        "httfatjet_regr_px", "httfatjet_regr_py", "httfatjet_regr_pz", "httfatjet_regr_e", 
        *[
            f"vbfjet{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e", "pnet_QvsG"
            ]
        ],
        "M_chi",
        "VBFjj_mass", "VBFdeltaR",
        "etaprod_bb", "etaprod_vbfjvbfj",
        "fwMoment_s_0", "fwMoment_T_0", "fwMoment_1_0",
        "fwMoment_s_2",
    ],
    "boosted": [
        "dau1_pt", "dau1_eta", "dau2_pt", "dau2_eta",
        "Htt_svfit_mass", "Htt_svfit_pt", "fatbjet_pt", "fatbjet_msoftdrop",
        "HH_svfit_mass_msoftdrop", "HH_svfit_pt", "HH_svfit_eta",
        "Htt_svfit_Hbb_softdrop_deltaEta", "Htt_svfit_Hbb_softdrop_deltaPhi", "dau1_dau2_deltaEta", "dau1_dau2_deltaPhi", 
        "Hbb_dau1_deltaR", "Hbb_dau2_deltaR"
    ]
}

cat_feature_sets = {
    "reg": [
        # order is important here since it is used as is for the tauNN
        "pairType", "dau1_DM", "dau2_DM", "dau1_charge", "dau2_charge",
    ],
    "default": [
        "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge", "isBoosted",
    ],
    "default_pnet": [
        "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge", "pass_pnet",
    ],
    "default_extended": [
        "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge", "isBoosted",
        "has_bjet1", "has_bjet2",
    ],
    "default_extended_pair": [
        "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge", "isBoosted",
        "has_bjet_pair",
    ],
    "full": (cat_features_full := [
        "pairType", "dau1_DM", "dau2_DM", "dau1_charge", "dau2_charge", "hasBoostedAK8",# "top_mass_idx",
    ]),
    "class": [
        "isBoosted", "pairType", "has_vbf_pair",
    ],
    "vbf": [
        "pairType", "dau1_DM", "dau2_DM", "dau1_charge", "dau2_charge", "hasResolvedAK4", "hasBoostedAK8", "hasVBFAK4",
    ],
    "boosted": [
        "dau1_DM", "dau2_DM", "dau1_charge", "dau2_charge", "hasBoostedAK8",
    ],
}

# selection sets can be strings, lists (which will be AND joined) or dictionaries with years mapping to strings or lists
# (in the latter case, the training script will choose the year automatically based on the sample)
selection_sets = {
    "baseline": (baseline_selection := [
        "n_btagged_jets > 1",
       # "nleps == 0",
        "isOS == 1",
        "PuppiMET_covXX >= 0",
        "PuppiMET_covYY >= 0",
        # # "fwMoment_eta_2 >= -1", 
        # # "fwMoment_eta_2 < 100",
        # # "relHtt_pt_mass < 100000", "relHbb_pt_mass < 100000",
        # # "dau2_tauIdVSjet >= 5",
        (
        "((pairType == 0) & (dau2_tauIdVSjet >= 5)) | "
        "((pairType == 1) & (dau2_tauIdVSjet >= 5)) | "
        "((pairType == 2) & (dau2_tauIdVSjet >= 5) & (dau1_tauIdVSjet >= 5)) | "
        "((pairType == 6) & (dau2_tauIdVSjet >= 0.984) & (dau1_tauIdVSjet >= 0.984) & (deltaRtautau > 0.05))"
        ),
    ]),
    "baseline_boosted": (baseline_selection := [
        "isOS == 1",
        "dau2_tauIdVSjet >= 5",
        (
            "(pairType == 0) | "
            "(pairType == 1) | "
            "((pairType == 2) & (dau1_tauIdVSjet >= 5))"
        ),
    ]),
    "baseline_lbtag": {
        year: baseline_selection + [
            f"(bjet1_bID_deepFlavor > {w['loose']}) | (bjet2_bID_deepFlavor > {w['loose']})",
        ]
        for year, w in btag_wps.items()
    },
    "signal": {
        year: baseline_selection + [
            (
                f"(bjet1_bID_deepFlavor > {w['medium']}) | "
                f"(bjet2_bID_deepFlavor > {w['medium']}) | "
                f"((isBoosted == 1) & (bjet1_bID_deepFlavor > {w['loose']}) & (bjet2_bID_deepFlavor > {w['loose']}))"
            ),
        ]
        for year, w in btag_wps.items()
    },
    "new_baseline": [
        #"nleps == 0",
        "isOS == 1",
        "dau2_deepTauVsJet >= 5",
        "((n_btagged_jets > 1) | (isBoosted == 1))",
        "((isLeptrigger == 1) | (isMETtrigger == 1) | (isSingleTautrigger == 1))",
        (
            "((pairType == 0) & (dau1_iso < 0.15)) | "
            "((pairType == 1) & (dau1_eleMVAiso == 1)) | "
            "((pairType == 2) & (dau1_deepTauVsJet >= 5))"
        ),
    ],
}

klub_aliases: dict[str, str] = {}

klub_index_columns = [
    "event",
    "run",
    "luminosityBlock",
]

klub_category_columns = []

klub_weight_columns = [
    "genWeight",
    "puWeight",
    # "trigSF",
    "DYstitchWeight",
    "idAndIsoAndFakeSF",
    "bTagweightReshape",
    "PrescaleWeight_PNetTauTau0p03",
]

klub_extra_weight_columns = []
klub_extra_columns = []

cclub_aliases: dict[str, str] = {
    "vbfjet1_px": "vbfjet1_px_nom",
    "vbfjet1_py": "vbfjet1_py_nom", 
    "vbfjet1_pz": "vbfjet1_pz_nom", 
    "vbfjet1_e": "vbfjet1_e_nom",
    "vbfjet2_px": "vbfjet2_px_nom",
    "vbfjet2_py": "vbfjet2_py_nom", 
    "vbfjet2_pz": "vbfjet2_pz_nom", 
    "vbfjet2_e": "vbfjet2_e_nom",
    "vbfjet1_pnet_QvsG": "vbfjet1_btagQvG",
    "vbfjet2_pnet_QvsG": "vbfjet1_btagQvG",
    "fatbjet_pt": "fatbjet_pt_nom",
    "fatbjet_pt": "fatbjet_pt_nom",
    "fatbjet_msoftdrop": "fatbjet_msoftdrop_nom",
    "vbfjet1_pt": "vbfjet1_pt_nom",
    "vbfjet2_pt": "vbfjet2_pt_nom",
    "fatjet_phi": "fatbjet_phi",
    "met_pt": "PuppiMET_smeared_pt",
    "met_phi": "PuppiMET_smeared_phi",
    "cH_bb": "cH_bb_nom",
    "relHbb_pt_mass": "relHbb_pt_mass_nom",
    "met_cov00": "PuppiMET_covXX",
    "met_cov01": "PuppiMET_covXY",
    "met_cov11": "PuppiMET_covYY",
    #"met_px": "puppimet_smeared_px",
    #"met_py": "puppimet_smeared_py",
    "VBFjj_deltaEta": "VBFjj_deltaEta_nom",
    "VBFjj_mass": "VBFjj_mass_nom",
    "VBFdeltaR":"VBFdeltaR_nom",
    "etaprod_vbfjvbfj": "etaprod_vbfjvbfj_nom",
    "deta_bb": "deta_bb_nom",
    "fwMoment_s_0": "fwMoment_s_0_nom",
    "fwMoment_p_0": "fwMoment_p_0_nom",
    "fwMoment_T_0": "fwMoment_T_0_nom",
    "fwMoment_z_0": "fwMoment_z_0_nom",
    "fwMoment_s_2": "fwMoment_s_2_nom",
    "fwMoment_p_2": "fwMoment_p_2_nom",
    "fwMoment_T_2": "fwMoment_T_2_nom",
    "fwMoment_z_2": "fwMoment_z_2_nom", 
    "fwMoment_s_3": "fwMoment_s_3_nom",
    "fwMoment_p_3": "fwMoment_p_3_nom",
    "fwMoment_T_3": "fwMoment_T_3_nom",
    "fwMoment_z_3": "fwMoment_z_3_nom", 
    "fwMoment_s_4": "fwMoment_s_4_nom",
    "fwMoment_p_4": "fwMoment_p_4_nom",
    "fwMoment_T_4": "fwMoment_T_4_nom",
    "fwMoment_z_4": "fwMoment_z_4_nom", 
    "fwMoment_s_8": "fwMoment_s_8_nom",
    "fwMoment_p_8": "fwMoment_p_8_nom",
    "fwMoment_T_8": "fwMoment_T_8_nom",
    "fwMoment_z_8": "fwMoment_z_8_nom",
    "fwMoment_s_0": "fwMoment_s_0_nom",
    "fwMoment_p_0": "fwMoment_p_0_nom",
    "fwMoment_T_0": "fwMoment_T_0_nom",
    "fwMoment_z_0": "fwMoment_z_0_nom",
    "fwMoment_s_2": "fwMoment_s_2_nom",
    "fwMoment_p_2": "fwMoment_p_2_nom",
    "fwMoment_T_2": "fwMoment_T_2_nom",
    "fwMoment_z_2": "fwMoment_z_2_nom", 
    "fwMoment_s_3": "fwMoment_s_3_nom",
    "fwMoment_p_3": "fwMoment_p_3_nom",
    "fwMoment_T_3": "fwMoment_T_3_nom",
    "fwMoment_z_3": "fwMoment_z_3_nom", 
    "fwMoment_s_4": "fwMoment_s_4_nom",
    "fwMoment_p_4": "fwMoment_p_4_nom",
    "fwMoment_T_4": "fwMoment_T_4_nom",
    "fwMoment_z_4": "fwMoment_z_4_nom", 
    "fwMoment_s_8": "fwMoment_s_8_nom",
    "fwMoment_p_8": "fwMoment_p_8_nom",
    "fwMoment_T_8": "fwMoment_T_8_nom",
    "fwMoment_z_8": "fwMoment_z_8_nom",
    "Hbb_pt": "Hbb_pt_nom",
    "Hbb_eta": "Hbb_eta_nom", 
    "Hbb_phi": "Hbb_phi_nom", 
    "Hbb_mass": "Hbb_mass_nom",
}

klub_index_columns = [
    "EventNumber",
    "RunNumber",
    "lumi",
]

klub_category_columns = [
    "pairType",
    "nleps",
    "isOS",
    "n_btagged_jets",
    "bjet1_bID_deepFlavor",
    "bjet2_bID_deepFlavor",
    "isBoosted",
    "isLeptrigger",
    "isMETtrigger",
    "isSingleTautrigger",
    "fatjet_particleNetMDJetTags_score",
    "fatjet_softdropMass",
    "dau1_iso",
    "dau1_eleMVAiso",
    "dau1_deepTauVsJet",
    "dau2_deepTauVsJet",
    "tauH_mass",
    "bH_mass",
    # preemptively add pt and eta values
    *[
        f"{obj}_{f}"
        for obj in ["dau1", "dau2", "bjet1", "bjet2", "fatjet"]
        for f in ["pt", "eta"]
    ],
]

cclub_weight_columns = [
    "genWeight",
    "puWeight",
    # "trigSF",
    # "DYstitchWeight",
    "idAndIsoAndFakeSF",
    "bTagweightReshape",
]

klub_extra_weight_columns = [
    "fatjet_particleNetMDJetTags_LP_SF",
]

reg_plot_columns = [
    "dau1_px", "dau1_py", "dau1_pz", "dau1_e",
    "dau2_px", "dau2_py", "dau2_pz", "dau2_e",
    "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e",
    "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e",
    "met_px", "met_py",
    "tauH_mass", "tauH_pt", "tauH_px", "tauH_py", "tauH_pz", "tauH_e",
    "tauH_SVFIT_mass", "tauH_SVFIT_pt", "tauH_SVFIT_px", "tauH_SVFIT_py", "tauH_SVFIT_pz", "tauH_SVFIT_e",
    "bH_px", "bH_py", "bH_pz", "bH_e",
    "HH_mass", "HH_pt",
    "svfit_HH_mass", "svfit_HH_pt",
    "recoGenTauH_pt", "recoGenTauH_mass",
    "genNu1_px", "genNu1_py", "genNu1_pz",
    "genNu2_px", "genNu2_py", "genNu2_pz",
    "recoGen_HH_pt", "recoGen_HH_mass",
]

dynamic_columns = {
    (rot_phi := "dau_phi"): (
        ("dau1_pt", "dau1_phi", "dau2_pt", "dau2_phi"),
        (lambda pt1, phi1, pt2, phi2: np.arctan2(
            pt1 * np.sin(phi1) + pt2 * np.sin(phi2),
            pt1 * np.cos(phi1) + pt2 * np.cos(phi2),
        )),
    ),
    "htt_regr_dphi": (
        ("Htt_regr_phi", rot_phi),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "htt_regr_px": (
        ("Htt_regr_pt", "htt_regr_dphi"),
        (lambda a, b: a * np.cos(b)),
    ),
    "htt_regr_py": (
        ("Htt_regr_pt", "htt_regr_dphi"),
        (lambda a, b: a * np.sin(b)),
    ),
    "htt_regr_pz": (
        ("Htt_regr_pt", "Htt_regr_eta"),
        (lambda a, b: a * np.sinh(b)),
    ),
    "htt_regr_e": (
        ("Htt_regr_pt", "Htt_regr_eta", "Htt_regr_phi", "Htt_regr_mass"),
        (lambda a, b, c, d: calc_energy(a, b, c, d)),
    ),
    "htthbb_regr_dphi": (
        ("htthbb_regr_phi", rot_phi),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "bjet1_px": (
        ("hasResolvedAK4", "bjet1_px_nom"),
        (lambda a, b: a * b),
    ),
    "bjet1_py": (
        ("hasResolvedAK4", "bjet1_py_nom"),
        (lambda a, b: a * b),
    ),
    "bjet1_pz": (
        ("hasResolvedAK4", "bjet1_pz_nom"),
        (lambda a, b: a * b),
    ),
    "bjet1_e": (
        ("hasResolvedAK4", "bjet1_e_nom"),
        (lambda a, b: a * b),
    ),
    "bjet1_pnet_b": (
        ("hasResolvedAK4", "bjet1_btag"),
        (lambda a, b: a * b + (-1) * (1 - a)),
    ),
    "bjet1_pnet_CvsB": (
        ("hasResolvedAK4", "bjet1_btagCvB"),
        (lambda a, b: a * b + (-1) * (1 - a)),
    ),
    "bjet1_pnet_CvsL": (
        ("hasResolvedAK4", "bjet1_btagCvL"),
        (lambda a, b: a * b + (-1) * (1 - a)),
    ),
    "bjet1_HHbtag": (
        ("hasResolvedAK4", "bjet1_hhbtag"),
        (lambda a, b: a * b + (-1) * (1 - a)),
    ),
    "bjet2_px": (
        ("hasResolvedAK4", "bjet2_px_nom"),
        (lambda a, b: a * b),
    ),
    "bjet2_py": (
        ("hasResolvedAK4", "bjet2_py_nom"),
        (lambda a, b: a * b),
    ),
    "bjet2_pz": (
        ("hasResolvedAK4", "bjet2_pz_nom"),
        (lambda a, b: a * b),
    ),
    "bjet2_e": (
        ("hasResolvedAK4", "bjet2_e_nom"),
        (lambda a, b: a * b),
    ),
    "bjet2_pnet_b": (
        ("hasResolvedAK4", "bjet2_btag"),
        (lambda a, b: a * b + (-1) * (1 - a)),
    ),
    "bjet2_pnet_CvsB": (
        ("hasResolvedAK4", "bjet2_btagCvB"),
        (lambda a, b: a * b + (-1) * (1 - a)),
    ),
    "bjet2_pnet_CvsL": (
        ("hasResolvedAK4", "bjet2_btagCvL"),
        (lambda a, b: a * b + (-1) * (1 - a)),
    ),
    "bjet2_HHbtag": (
        ("hasResolvedAK4", "bjet2_hhbtag"),
        (lambda a, b: a * b + (-1) * (1 - a)),
    ),
    "hbb_px": (
        ("hasResolvedAK4", "hbb_px_nom"),
        (lambda a, b: a * b),
    ),
    "hbb_py": (
        ("hasResolvedAK4", "hbb_py_nom"),
        (lambda a, b: a * b),
    ),
    "hbb_pz": (
        ("hasResolvedAK4", "hbb_pz_nom"),
        (lambda a, b: a * b),
    ),
    "hbb_e": (
        ("hasResolvedAK4", "hbb_e_nom"),
        (lambda a, b: a * b),
    ),
    "htthbb_regr_px": (
        ("hasResolvedAK4", "htt_regr_px", "hbb_px_nom"),
        (lambda a, b, c: a * (b + c)),
    ),
    "htthbb_regr_py": (
        ("hasResolvedAK4", "htt_regr_py", "hbb_py_nom"),
        (lambda a, b, c: a * (b + c)),
    ),
    "htthbb_regr_pz": (
        ("hasResolvedAK4", "htt_regr_pz", "hbb_pz_nom"),
        (lambda a, b, c: a * (b + c)),
    ),
    "htthbb_regr_e": (
        ("hasResolvedAK4", "htt_regr_e", "hbb_e_nom"),
        (lambda a, b, c: a * (b + c)),
    ),
    "fatjet_e": (
        ("hasBoostedAK8", "fatbjet_e_nom"),
        (lambda a, b: a * b),
    ),
    "fatjet_px": (
        ("hasBoostedAK8", "fatbjet_px_nom"),
        (lambda a, b: a * b),
    ),
    "fatjet_py": (
        ("hasBoostedAK8", "fatbjet_py_nom"),
        (lambda a, b: a * b),
    ),
    "fatjet_pz": (
        ("hasBoostedAK8", "fatbjet_pz_nom"),
        (lambda a, b: a * b),
    ),
    "httfatjet_regr_pz": (
        ("hasBoostedAK8", "htt_regr_pz", "fatjet_pz"),
        (lambda a, b, c: a * (b + c)),
    ),
    "httfatjet_regr_e": (
        ("hasBoostedAK8", "htt_regr_e", "fatjet_e"),
        (lambda a, b, c: a * (b + c)),
    ),
     "httfatjet_regr_px": (
        ("hasBoostedAK8","htt_regr_px", "fatjet_px"),
        (lambda a, b, c: a * (b + c)),
    ),
    "httfatjet_regr_py": (
        ("hasBoostedAK8", "htt_regr_py", "fatjet_py"),
        (lambda a, b, c: a * (b + c)),
    ),
    "M_chi": (
        ("hasResolvedAK4", "HH_regr_mass", "Htt_regr_mass", "Hbb_mass_nom"),
        (lambda d, a, b, c: d * (a - (b - 125.0) - (c - 125.0))),
    ),
    "etaprod_bb": (
        ("hasResolvedAK4", "etaprod_bb_nom"),
        (lambda a, b: a * b)
    ),
    "met_dphi": (
        ("PuppiMET_phi", rot_phi),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "met_px": (
        ("PuppiMET_pt", "met_dphi"),
        (lambda a, b: a * np.cos(b)),
    ),
    "met_py": (
        ("PuppiMET_pt", "met_dphi"),
        (lambda a, b: a * np.sin(b)),
    ),
    # "mask_fatbjet":(
    #     ("fatbjet_pt_nom"),
    #     (lambda a: a>0)
    # ),
    "fatjet_masked_e":(
        ("fatjet_e", "fatbjet_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "fatjet_masked_px":(
        ("fatjet_px", "fatbjet_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "fatjet_masked_py":(
        ("fatjet_py", "fatbjet_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "fatjet_masked_pz":(
        ("fatjet_pz", "fatbjet_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    # "mask_bjet1":(
    #     ("bjet1_pt_nom"),
    #     (lambda a: a>0)
    # ),
    "bjet1_masked_e":(
        ("bjet1_e", "bjet1_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "bjet1_masked_px":(
        ("bjet1_px", "bjet1_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "bjet1_masked_py":(
        ("bjet1_py", "bjet1_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "bjet1_masked_pz":(
        ("bjet1_pz", "bjet1_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "bjet1_masked_btagDeepFlavB":(
        ("bjet1_btagDeepFlavB", "bjet1_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "bjet1_masked_PNetCvB": (
        ("bjet1_PNetCvB", "bjet1_pt_nom"),
        (lambda a, b: a * (b>0))
    ), 
    "bjet1_masked_PNetCvL":(
        ("bjet1_PNetCvL", "bjet1_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "bjet1_masked_HHbtag":(
        ("bjet1_HHbtag", "bjet1_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    # "mask_bjet2":(
    #     ("bjet2_pt_nom"),
    #     (lambda a: a>0)
    # ),
    "bjet2_masked_e":(
        ("bjet2_e", "bjet2_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "bjet2_masked_px":(
        ("bjet2_px", "bjet2_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "bjet2_masked_py":(
        ("bjet2_py", "bjet2_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "bjet2_masked_pz":(
        ("bjet2_pz", "bjet2_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "bjet2_masked_btagDeepFlavB":(
        ("bjet2_btagDeepFlavB", "bjet2_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "bjet2_masked_PNetCvB": (
        ("bjet2_PNetCvB", "bjet2_pt_nom"),
        (lambda a, b: a * (b>0))
    ), 
    "bjet2_masked_PNetCvL":(
        ("bjet2_PNetCvL", "bjet2_pt_nom"),
        (lambda a, b: a * (b>0))
    ),
    "bjet2_masked_HHbtag":(
        ("bjet2_HHbtag", "bjet2_pt_nom"),
        (lambda a, b: a * (b>0))
    )
}



embedding_expected_inputs = {
    "pairType": [0, 1, 2, 3, 4, 5, 6],
    "dau1_DM": [-999, 0, 1, 10, 11],  # -1 for e/mu
    "dau2_DM": [-999, 0, 1, 10, 11],
    "dau1_charge": [-1, 1],
    "dau2_charge": [-1, 1],
    # "spin": [0, 2],
    # "year": [0, 1, 2, 3, 4, 5, 6, 7, 8,],
    # "hasResolvedAK4": [0, 1],
    "hasBoostedAK8": [0, 1],
    # "hasVBFAK4": [0, 1],
    # "pass_pnet": [0, 1],
    #"top_mass_idx": [0, 1, 2, 3],
    # "has_bjet1": [0, 1],
    # "has_bjet2": [0, 1],
    # "has_bjet_pair": [0, 1],
}


@dataclass
class RegressionSet:

    model_files: dict[int, str]
    cont_feature_set: str
    cat_feature_set: str
    parameterize_year: bool = False
    parameterize_spin: bool = True
    parameterize_mass: bool = True
    use_reg_outputs: bool = True
    use_reg_last_layer: bool = True
    use_cls_outputs: bool = True
    use_cls_last_layer: bool = True
    fadein: tuple[int, int] = (0, 0)  # fade-in start, duration
    fine_tune: dict[str, Any] | None = None
    feed_lbn: bool = False

    def copy(self, **attrs) -> RegressionSet:
        kwargs = self.__dict__.copy()
        kwargs.update(attrs)
        return self.__class__(**kwargs)


regression_sets = {
    "default": (default_reg_set := RegressionSet(
        model_files={
            fold: os.path.join(os.getenv("TN_REG_MODEL_DIR"), "RegTraining/reg_mass_para_class_l2n400_removeMoreVars_addBkgs_addlast_set_1")  # noqa
            for fold in range(10)
        },
        cont_feature_set="reg2",
        cat_feature_set="reg",
        parameterize_year=False,
        parameterize_spin=True,
        parameterize_mass=True,
        use_reg_outputs=False,
        use_reg_last_layer=True,
        use_cls_outputs=False,
        use_cls_last_layer=True,
        fadein=(150, 20),
        fine_tune=None,
        feed_lbn=False,
    )),
    "v2": (reg_set_v2 := RegressionSet(
        model_files={
            fold: os.path.join(os.getenv("TN_REG_MODEL_DIR"), "RegTraining/ttreg_ED5_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadam_LR3.0e-03_YEARy_SPINy_MASSy_FI0_SD1")  # noqa
            for fold in range(10)
        },
        cont_feature_set="reg_v2",
        cat_feature_set="default",
        parameterize_year=True,
        parameterize_spin=True,
        parameterize_mass=True,
        use_reg_outputs=False,
        use_reg_last_layer=True,
        use_cls_outputs=False,
        use_cls_last_layer=True,
        fadein=(150, 20),
        fine_tune=None,
        feed_lbn=False,
    )),
    "v2_ft": reg_set_v2.copy(
        fine_tune={
            # use same norm as dnn
            "l2_norm": lambda dnn_l2_norm: dnn_l2_norm * 4,
            # use current learning rate, but with two reverse reduction steps
            "learning_rate": lambda dnn_initial_lr, current_lr: current_lr * 0.5**-2,
        },
    ),
    # "v2_lbn": reg_set_v2.copy(feed_lbn=True),
    # "v2_lbn_passall": reg_set_v2.copy(feed_lbn=True, use_reg_outputs=True, use_cls_outputs=True),
    "v3": (reg_set_v3 := RegressionSet(
        model_files={
            fold: os.path.join(os.getenv("TN_STORE_DIR_TOBI"), f"RegTraining/daurot_v3/tautaureg_PSbaseline_LSmulti4_SSdefault_FSdefault_daurot-default_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadam_LR3.0e-03_YEARy_SPINy_MASSy_FI{fold}_SD1_val_metric_sum")  # noqa
            for fold in range(5)
        },
        cont_feature_set="default_daurot",
        cat_feature_set="default",
        parameterize_year=True,
        parameterize_spin=True,
        parameterize_mass=True,
        use_reg_outputs=False,
        use_reg_last_layer=True,
        use_cls_outputs=False,
        use_cls_last_layer=True,
        fadein=(150, 20),
        fine_tune=None,
        feed_lbn=False,
    )),
    "v3_lbn": reg_set_v3.copy(feed_lbn=True),
    "v3_lbn_ft_lt10_lr2": reg_set_v3.copy(
        feed_lbn=True,
        fine_tune={
            "l2_norm": lambda dnn_l2_norm: dnn_l2_norm * 10,
            "learning_rate": lambda dnn_initial_lr, current_lr: current_lr * 0.5**-2,
        },
    ),
    "v3_lbn_ft_lt20_lr1": reg_set_v3.copy(
        feed_lbn=True,
        fine_tune={
            "l2_norm": lambda dnn_l2_norm: dnn_l2_norm * 20,
            "learning_rate": lambda dnn_initial_lr, current_lr: current_lr * 0.5**-1,
        },
    ),
    "v3_ft_lt20_lr1": reg_set_v3.copy(
        fine_tune={
            "l2_norm": lambda dnn_l2_norm: dnn_l2_norm * 20,
            "learning_rate": lambda dnn_initial_lr, current_lr: current_lr * 0.5**-1,
        },
    ),
    "v4_1fold": (reg_set_v4pre := RegressionSet(
        model_files={
            # just one fold
            0: os.path.join(os.getenv("TN_STORE_DIR_TOBI"), "RegTraining/new_skims_all_samples/tautaureg_PSnew_baseline_LSmulti4_SSdefault_FSdefault_daurot_fatjet-default_pnet_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR3.0e-03_YEARy_SPINy_MASSy_FI0_SD1")  # noqa
        },
        cont_feature_set="default_daurot_fatjet",
        cat_feature_set="default_pnet",
        parameterize_year=True,
        parameterize_spin=True,
        parameterize_mass=True,
        use_reg_outputs=False,
        use_reg_last_layer=True,
        use_cls_outputs=False,
        use_cls_last_layer=True,
        fadein=(150, 20),
        fine_tune=None,
        feed_lbn=False,
    )),
    "v6": (reg_set_v6 := RegressionSet(
        model_files={
            fold: os.path.join(os.getenv("TN_STORE_DIR_TOBI"), f"RegTraining/dev_final_features3/tautaureg_PSnew_baseline_LSmulti4_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR3.0e-03_YEARy_SPINy_MASSy_FI{fold}_SD1")  # noqa
            for fold in range(5)
        },
        cont_feature_set="default_daurot_composite",
        cat_feature_set="default_extended_pair",
        parameterize_year=True,
        parameterize_spin=True,
        parameterize_mass=True,
        use_reg_outputs=False,
        use_reg_last_layer=True,
        use_cls_outputs=False,
        use_cls_last_layer=True,
        fadein=(150, 20),
        fine_tune=None,
        feed_lbn=False,
    )),
    "v6_lbn_ft_lt20_lr1": (reg_set_v6_lbn_ft_lt20_lr1 := reg_set_v6.copy(
        feed_lbn=True,
        fine_tune={
            "l2_norm": lambda dnn_l2_norm: dnn_l2_norm * 20,
            "learning_rate": lambda dnn_initial_lr, current_lr: current_lr * 0.5**-1,
        },
    )),
    "v6_fi80_lbn_ft_lt20_lr1": reg_set_v6_lbn_ft_lt20_lr1.copy(
        fadein=(80, 20),
    ),
    "v6_fi100_lbn_ft_lt20_lr1": reg_set_v6_lbn_ft_lt20_lr1.copy(
        fadein=(100, 20),
    ),
}


@dataclass
class LBNSet:

    input_features: list[str | None]
    output_features: list[str]
    boost_mode: str
    n_particles: int
    n_restframes: int | None = None

    def copy(self, **attrs) -> LBNSet:
        kwargs = self.__dict__.copy()
        kwargs.update(attrs)
        return self.__class__(**kwargs)


lbn_sets = {
    "test": LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            None, "met_px", "met_py", None,
            None, "dmet_resp_px", "dmet_resp_py", None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos"],
        boost_mode="pairs",
        n_particles=7,
    ),
    "test2": LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            "tauH_e", "tauH_px", "tauH_py", "tauH_pz",
            "bH_e", "bH_px", "bH_py", "bH_pz",
            None, "met_px", "met_py", None,
            None, "dmet_resp_px", "dmet_resp_py", None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos"],
        boost_mode="pairs",
        n_particles=7,
    ),
    "test3": (lbn_test3 := LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            "tauH_e", "tauH_px", "tauH_py", "tauH_pz",
            "bH_e", "bH_px", "bH_py", "bH_pz",
            "HH_e", "HH_px", "HH_py", "HH_pz",
            None, "met_px", "met_py", None,
            None, "dmet_resp_px", "dmet_resp_py", None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos"],
        boost_mode="pairs",
        n_particles=7,
    )),
    "test4": (lbn_test4 := LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            "tauH_e", "tauH_px", "tauH_py", "tauH_pz",
            "bH_e", "bH_px", "bH_py", "bH_pz",
            "HH_e", "HH_px", "HH_py", "HH_pz",
            None, "met_px", "met_py", None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos", "pair_dr"],
        boost_mode="pairs",
        n_particles=7,
    )),
    "test4_metfix": (lbn_test4_metfix := LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            "tauH_e", "tauH_px", "tauH_py", "tauH_pz",
            "bH_e", "bH_px", "bH_py", "bH_pz",
            "HH_e", "HH_px", "HH_py", "HH_pz",
            None, "met_et", None, None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos", "pair_dr"],
        boost_mode="pairs",
        n_particles=7,
    )),
    "test5": lbn_test4_metfix.copy(boost_mode="product", n_restframes=4),
    "default_metrot": LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            "tauH_e", "tauH_px", "tauH_py", "tauH_pz",
            "bH_e", "bH_px", "bH_py", "bH_pz",
            "HH_e", "HH_px", "HH_py", "HH_pz",
            None, "met_et", None, None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos"],
        boost_mode="pairs",
        n_particles=8,
    ),
    "default_daurot": LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            "tauH_e", "tauH_px", "tauH_py", "tauH_pz",
            "bH_e", "bH_px", "bH_py", "bH_pz",
            "HH_e", "HH_px", "HH_py", "HH_pz",
            None, "met_px", "met_py", None,
            None, "dmet_resp_px", "dmet_resp_py", None,
            None, "dmet_reso_px", "dmet_reso_py", None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos"],
        boost_mode="pairs",
        n_particles=10,
    ),
    "vbf": LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            "vbfjet1_e", "vbfjet1_px", "vbfjet1_py", "vbfjet1_pz",
            "vbfjet2_e", "vbfjet2_px", "vbfjet2_py", "vbfjet2_pz",
            "fatjet_e", "fatjet_px", "fatjet_py", "fatjet_pz",
            None, "nu1_px", "nu1_py", "nu1_pz",
            None, "nu2_px", "nu2_py", "nu2_pz",
            "htt_regr_e", "htt_regr_px", "htt_regr_py", "htt_regr_pz",
            "hbb_e", "hbb_px", "hbb_py", "hbb_pz",
            "htthbb_regr_e", "htthbb_regr_px", "htthbb_regr_py", "htthbb_regr_pz",
            "httfatjet_regr_e", "httfatjet_regr_px", "httfatjet_regr_py", "httfatjet_regr_pz",
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos"],
        boost_mode="pairs",
        n_particles=10,
    ),
    "default_daurot_fatjet": LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_masked_e", "bjet1_masked_px", "bjet1_masked_py", "bjet1_masked_pz",
            "bjet2_masked_e", "bjet2_masked_px", "bjet2_masked_py", "bjet2_masked_pz",
            "fatjet_masked_e", "fatjet_masked_px", "fatjet_masked_py", "fatjet_masked_pz",
            # "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            # "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            # "fatjet_e", "fatjet_px", "fatjet_py", "fatjet_pz",
            None, "met_px", "met_py", None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos"],
        boost_mode="pairs",
        n_particles=10,
    ),
    "default_daurot_fatjet_composite": LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_masked_e", "bjet1_masked_px", "bjet1_masked_py", "bjet1_masked_pz",
            "bjet2_masked_e", "bjet2_masked_px", "bjet2_masked_py", "bjet2_masked_pz",
            "fatjet_masked_e", "fatjet_masked_px", "fatjet_masked_py", "fatjet_masked_pz",
            "htt_e", "htt_px", "htt_py", "htt_pz",
            "htthbb_masked_e", "htthbb_masked_px", "htthbb_masked_py", "htthbb_masked_pz",
            "httfatjet_masked_e", "httfatjet_masked_px", "httfatjet_masked_py", "httfatjet_masked_pz",
            None, "met_px", "met_py", None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos"],
        boost_mode="pairs",
        n_particles=10,
    ),
}
