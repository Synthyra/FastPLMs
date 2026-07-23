import math

import numpy as np
import torch
from torch.nn.functional import one_hot

from . import vb_const as const
from .minimal_structures import ProteinStructureTemplate

_ELEMENT_TO_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "P": 15,
    "S": 16,
}

# Canonical formal charges and tetrahedral chirality values extracted from the
# pinned Boltz2 molecule archive. Values use Boltz2's public feature encoding.
_FORMAL_CHARGES = {
    ("ARG", "NH1"): 1.0,
    ("HIS", "ND1"): 1.0,
    ("LYS", "NZ"): 1.0,
}
_CHIRAL_ATOMS = {
    "ALA": frozenset({"CA"}),
    "ARG": frozenset({"CA"}),
    "ASN": frozenset({"CA"}),
    "ASP": frozenset({"CA"}),
    "CYS": frozenset({"CA"}),
    "GLN": frozenset({"CA"}),
    "GLU": frozenset({"CA"}),
    "HIS": frozenset({"CA"}),
    "ILE": frozenset({"CA", "CB"}),
    "LEU": frozenset({"CA"}),
    "LYS": frozenset({"CA"}),
    "MET": frozenset({"CA"}),
    "PHE": frozenset({"CA"}),
    "PRO": frozenset({"CA"}),
    "SER": frozenset({"CA"}),
    "THR": frozenset({"CA", "CB"}),
    "TRP": frozenset({"CA"}),
    "TYR": frozenset({"CA"}),
    "UNK": frozenset({"CA"}),
    "VAL": frozenset({"CA"}),
}


def _normalize_sequence(sequence: str) -> str:
    if not isinstance(sequence, str):
        raise TypeError("Amino acid sequence must be a string.")
    seq = sequence.strip().upper()
    if not seq:
        raise ValueError("Amino acid sequence must be non-empty.")
    for aa in seq:
        if aa not in const.prot_letter_to_token:
            raise ValueError(f"Unsupported residue code '{aa}'.")
    return seq


def _atom_name_to_element(atom_name: str) -> str:
    name = atom_name.strip().upper()
    if len(name) == 0:
        return "C"
    if name[0].isdigit():
        name = name[1:]
    if len(name) >= 2 and name[0:2] in ("CL", "BR", "FE", "MG", "ZN", "NA", "CA"):
        return name[0]
    return name[0]


def _atom_name_to_codes(atom_name: str) -> torch.Tensor:
    clipped = atom_name.strip()[:4]
    vals = [ord(ch) - 32 for ch in clipped]
    while len(vals) < 4:
        vals.append(0)
    out = torch.tensor(vals, dtype=torch.long)
    if not (torch.all(out >= 0) and torch.all(out < 64)):
        raise ValueError(f"Invalid atom-name encoding for '{atom_name}'.")
    return out


# Raw first-conformer coordinates extracted at full float32 precision from the
# hash-pinned Boltz2 molecule archive. The feature builder applies the official
# seeded centering, rotation, and translation policy below.
_RDKIT_CONFORMERS: dict[str, dict[str, list[float]]] = {
    "ALA": {
        "N": [-0.9241582155227661, 1.1821246147155762, 0.712748110294342],
        "CA": [-0.2663755416870117, -0.08827890455722809, 0.4008508622646332],
        "C": [1.1188693046569824, 0.1387452930212021, -0.14366498589515686],
        "O": [1.2882297039031982, 0.8058497309684753, -1.2001261711120605],
        "CB": [-1.1134333610534668, -0.8915302753448486, -0.5876986384391785],
    },
    "ARG": {
        "N": [3.253326416015625, -1.699564814567566, -0.9852627515792847],
        "CA": [2.191511392593384, -0.692007839679718, -0.9506462812423706],
        "C": [2.80021333694458, 0.672307550907135, -1.1160222291946411],
        "O": [2.465440034866333, 1.3987600803375244, -2.0899009704589844],
        "CB": [1.3811039924621582, -0.7949594855308533, 0.36316436529159546],
        "CG": [0.20344306528568268, 0.19287171959877014, 0.457282155752182],
        "CD": [-0.902410626411438, -0.09953747689723969, -0.5671625137329102],
        "NE": [-2.0867979526519775, 0.7146661281585693, -0.2911669909954071],
        "CZ": [-3.018158197402954, 0.4804283082485199, 0.7831193804740906],
        "NH1": [-4.080137729644775, 1.4154621362686157, 0.9938872456550598],
        "NH2": [-2.9226150512695312, -0.5637124180793762, 1.5533623695373535],
    },
    "ASN": {
        "N": [-1.5767980813980103, -1.7227835655212402, 0.2427234649658203],
        "CA": [-0.7041534781455994, -0.5622008442878723, 0.42767149209976196],
        "C": [-1.246688961982727, 0.6072975993156433, -0.3483567237854004],
        "O": [-1.554918885231018, 0.47644123435020447, -1.5638835430145264],
        "CB": [0.7263866662979126, -0.9032993316650391, -0.028588544577360153],
        "CG": [1.6871562004089355, 0.20810098946094513, 0.28412675857543945],
        "OD1": [2.19687557220459, 0.28916797041893005, 1.433493971824646],
        "ND2": [1.996717095375061, 1.196706771850586, -0.6981719136238098],
    },
    "ASP": {
        "N": [-0.11774874478578568, -1.6310220956802368, 0.374360054731369],
        "CA": [-0.3847021758556366, -0.19290897250175476, 0.28121331334114075],
        "C": [-1.806337594985962, 0.059408850967884064, -0.1472136378288269],
        "O": [-2.2237648963928223, -0.36817988753318787, -1.257502555847168],
        "CB": [0.5981687307357788, 0.4785478413105011, -0.6920451521873474],
        "CG": [2.0088446140289307, 0.36018264293670654, -0.20304732024669647],
        "OD1": [2.747267961502075, -0.5711832046508789, -0.6216948628425598],
        "OD2": [2.49090313911438, 1.2577818632125854, 0.7443997859954834],
    },
    "CYS": {
        "N": [-0.058702241629362106, 1.771048665046692, 0.2434082329273224],
        "CA": [-0.06546340137720108, 0.47663936018943787, -0.4456723928451538],
        "C": [-1.273919701576233, -0.3504071533679962, -0.08361759781837463],
        "O": [-1.7324607372283936, -0.34313708543777466, 1.0909277200698853],
        "CB": [1.2432398796081543, -0.28917407989501953, -0.19254730641841888],
        "SG": [1.484184980392456, -0.7191365361213684, 1.5649693012237549],
    },
    "GLN": {
        "N": [-1.8543237447738647, -1.0024770498275757, -1.6278940439224243],
        "CA": [-1.292184591293335, -0.6786512732505798, -0.3153715133666992],
        "C": [-2.2264492511749268, 0.25171151757240295, 0.4067918062210083],
        "O": [-2.7255098819732666, -0.08834805339574814, 1.5129321813583374],
        "CB": [0.11662524193525314, -0.06213820353150368, -0.45663803815841675],
        "CG": [0.8140791058540344, 0.10461423546075821, 0.9002283811569214],
        "CD": [2.195122003555298, 0.6569428443908691, 0.7154991030693054],
        "OE1": [2.3870291709899902, 1.9003654718399048, 0.7825236916542053],
        "NE2": [3.287371873855591, -0.21307705342769623, 0.41959717869758606],
    },
    "GLU": {
        "N": [-1.3492857217788696, -1.114271640777588, -1.3739068508148193],
        "CA": [-1.2765676975250244, -0.506334662437439, -0.04287439212203026],
        "C": [-1.9195666313171387, 0.8547031879425049, -0.04214814677834511],
        "O": [-1.9091347455978394, 1.5701956748962402, -1.080504298210144],
        "CB": [0.18272103369235992, -0.43209096789360046, 0.45213374495506287],
        "CG": [1.1058456897735596, 0.3659400939941406, -0.481784850358963],
        "CD": [2.5090205669403076, 0.35135790705680847, 0.03646872192621231],
        "OE1": [2.896414041519165, 1.2445733547210693, 0.8365193605422974],
        "OE2": [3.3727896213531494, -0.6754258275032043, -0.33141785860061646],
    },
    "GLY": {
        "N": [-1.291549801826477, 0.6080796122550964, -0.4228580892086029],
        "CA": [-0.4895951449871063, -0.2882237136363983, 0.40191715955734253],
        "C": [0.9350062608718872, -0.2543502449989319, -0.04786944016814232],
        "O": [1.3473576307296753, -1.0836502313613892, -0.9022219777107239],
    },
    "HIS": {
        "N": [1.0371263027191162, -1.5621215105056763, 0.4178937077522278],
        "CA": [1.1980191469192505, -0.40241551399230957, -0.46312251687049866],
        "C": [2.6561269760131836, -0.07123222947120667, -0.6407948136329651],
        "O": [3.1795547008514404, -0.12371637672185898, -1.7859554290771484],
        "CB": [0.44051393866539, 0.8116919994354248, 0.09850303083658218],
        "CG": [-1.0356801748275757, 0.5519806742668152, 0.15254715085029602],
        "ND1": [-1.9089158773422241, 0.6490958333015442, -0.968245267868042],
        "CD2": [-1.7095766067504883, 0.1138235554099083, 1.2061808109283447],
        "CE1": [-3.0751633644104004, 0.28129827976226807, -0.5511555075645447],
        "NE2": [-3.0650453567504883, -0.07846951484680176, 0.8257263898849487],
    },
    "ILE": {
        "N": [-1.2378733158111572, -1.8146690130233765, -0.16437499225139618],
        "CA": [-1.2780015468597412, -0.3994033634662628, 0.22437156736850739],
        "C": [-1.9879958629608154, 0.4497596025466919, -0.80372154712677],
        "O": [-2.0291709899902344, 0.10288607329130173, -2.015406608581543],
        "CB": [0.14225825667381287, 0.146738663315773, 0.5415176153182983],
        "CG1": [1.1062071323394775, 0.04519597440958023, -0.6694174408912659],
        "CG2": [0.7215384244918823, -0.5685611963272095, 1.7750537395477295],
        "CD1": [2.371805191040039, 0.8852734565734863, -0.49045464396476746],
    },
    "LEU": {
        "N": [1.6452248096466064, -1.017622470855713, -1.0626215934753418],
        "CA": [1.3129304647445679, 0.08074887096881866, -0.15053576231002808],
        "C": [2.431154727935791, 0.28310757875442505, 0.8357383608818054],
        "O": [2.9231173992156982, -0.7013152837753296, 1.451209306716919],
        "CB": [0.006050171796232462, -0.2141101360321045, 0.6198461651802063],
        "CG": [-1.2534235715866089, -0.3772832751274109, -0.26637399196624756],
        "CD1": [-2.455688714981079, -0.7604416012763977, 0.6070646643638611],
        "CD2": [-1.5742294788360596, 0.8985711932182312, -1.059889554977417],
    },
    "LYS": {
        "N": [-2.3938536643981934, -1.4751482009887695, -0.9336304068565369],
        "CA": [-2.2094509601593018, -0.6203567981719971, 0.2421264797449112],
        "C": [-3.4513463973999023, 0.1961250752210617, 0.4710204303264618],
        "O": [-3.9688456058502197, 0.2503468096256256, 1.6188287734985352],
        "CB": [-1.008726716041565, 0.3269270658493042, 0.0479682981967926],
        "CG": [0.336457222700119, -0.41559070348739624, 0.047045741230249405],
        "CD": [1.5088860988616943, 0.5716065168380737, -0.00632853340357542],
        "CE": [2.853410005569458, -0.1678532361984253, -0.020898228511214256],
        "NZ": [3.9690465927124023, 0.7765668630599976, -0.07007527351379395],
    },
    "MET": {
        "N": [-1.743452548980713, -0.44039490818977356, 1.851780652999878],
        "CA": [-1.2168668508529663, 0.4041755199432373, 0.7779996991157532],
        "C": [-2.3118340969085693, 0.683431088924408, -0.2134547382593155],
        "O": [-2.6660380363464355, 1.8705639839172363, -0.44476020336151123],
        "CB": [-0.0005482881097123027, -0.262584388256073, 0.09933014959096909],
        "CG": [0.6952047944068909, 0.6768104434013367, -0.8930339813232422],
        "SD": [2.136805772781372, -0.14913176000118256, -1.6563159227371216],
        "CE": [3.3404407501220703, 0.348065584897995, -0.38104408979415894],
    },
    "PHE": {
        "N": [2.7976438999176025, -1.6209964752197266, -0.4779351055622101],
        "CA": [1.5799994468688965, -0.8289614915847778, -0.6608061194419861],
        "C": [1.9535551071166992, 0.5758142471313477, -1.0445665121078491],
        "O": [1.5211102962493896, 1.072593331336975, -2.1190123558044434],
        "CB": [0.7137221097946167, -0.8474348187446594, 0.6163292527198792],
        "CG": [-0.6229045987129211, -0.18666480481624603, 0.3912639319896698],
        "CD1": [-0.8259273767471313, 1.0978055000305176, 0.7381877303123474],
        "CD2": [-1.7160141468048096, -0.9425657391548157, -0.2700555920600891],
        "CE1": [-2.131312131881714, 1.741182565689087, 0.4800238013267517],
        "CE2": [-2.8960046768188477, -0.3551671802997589, -0.505269467830658],
        "CZ": [-3.11456298828125, 1.0511738061904907, -0.11009909957647324],
    },
    "PRO": {
        "N": [-0.577339768409729, -0.5123927593231201, -1.150407314300537],
        "CA": [0.5119893550872803, 0.23565764725208282, -0.5195383429527283],
        "C": [1.8393173217773438, -0.46712079644203186, -0.6550371646881104],
        "O": [2.0836586952209473, -1.166333556175232, -1.6751221418380737],
        "CB": [0.1029478907585144, 0.45346882939338684, 0.9297074675559998],
        "CG": [-1.4098080396652222, 0.5215330123901367, 0.8787456750869751],
        "CD": [-1.7971025705337524, -0.06041542813181877, -0.47876954078674316],
    },
    "SER": {
        "N": [0.8959873914718628, -1.423298716545105, -0.2692987024784088],
        "CA": [-0.018787339329719543, -0.28751760721206665, -0.3863537311553955],
        "C": [0.767615795135498, 0.9929561018943787, -0.42147380113601685],
        "O": [0.5327092409133911, 1.8535151481628418, -1.3116248846054077],
        "CB": [-0.9973848462104797, -0.2769756019115448, 0.7988247871398926],
        "OG": [-1.8999865055084229, 0.7915746569633484, 0.691534698009491],
    },
    "THR": {
        "N": [0.005045319441705942, 1.7655576467514038, 0.0639338344335556],
        "CA": [0.4231603145599365, 0.3755212128162384, 0.2734847664833069],
        "C": [1.7184523344039917, 0.11012311279773712, -0.4465082287788391],
        "O": [1.9252793788909912, 0.5967074036598206, -1.5912789106369019],
        "CB": [-0.6634359359741211, -0.615561306476593, -0.21173778176307678],
        "OG1": [-1.0762907266616821, -0.29800114035606384, -1.5171915292739868],
        "CG2": [-1.8869022130966187, -0.6176311373710632, 0.7084044218063354],
    },
    "TRP": {
        "N": [-3.0439648628234863, 1.0628471374511719, 0.015692168846726418],
        "CA": [-2.463679552078247, 0.024657616391777992, -0.8414031863212585],
        "C": [-2.186711072921753, -1.2228965759277344, -0.04691409692168236],
        "O": [-2.3364906311035156, -2.3535354137420654, -0.5830773115158081],
        "CB": [-1.2062920331954956, 0.5195542573928833, -1.592403531074524],
        "CG": [-0.06450498104095459, 0.884079098701477, -0.6835198402404785],
        "CD1": [0.16109998524188995, 2.085163116455078, -0.14801783859729767],
        "CD2": [0.9781660437583923, -0.005544683896005154, -0.17733941972255707],
        "NE1": [1.3106837272644043, 2.0506503582000732, 0.698072075843811],
        "CE2": [1.7528777122497559, 0.7044392228126526, 0.6196392774581909],
        "CE3": [1.236093521118164, -1.4355946779251099, -0.4235689342021942],
        "CZ2": [2.9070138931274414, 0.10196790844202042, 1.3070299625396729],
        "CZ3": [2.2851555347442627, -2.007185697555542, 0.1949695348739624],
        "CH2": [3.1585631370544434, -1.2034809589385986, 1.1002938747406006],
    },
    "TYR": {
        "N": [-1.7950150966644287, 0.49119046330451965, -1.3951144218444824],
        "CA": [-1.8436503410339355, -0.2694968581199646, -0.14338290691375732],
        "C": [-3.240288019180298, -0.28215137124061584, 0.41974759101867676],
        "O": [-3.8150112628936768, 0.7977103590965271, 0.7253130674362183],
        "CB": [-0.8541494011878967, 0.30990728735923767, 0.8830214738845825],
        "CG": [0.5672639012336731, 0.2125871181488037, 0.3916495442390442],
        "CD1": [1.2694358825683594, -0.924761176109314, 0.5431669354438782],
        "CD2": [1.1872504949569702, 1.3605931997299194, -0.31554359197616577],
        "CE1": [2.6519992351531982, -1.0213464498519897, 0.029911600053310394],
        "CE2": [2.4384164810180664, 1.2693941593170166, -0.7831515073776245],
        "CZ": [3.211331367492676, 0.022552935406565666, -0.6021429300308228],
        "OH": [4.513507843017578, -0.05260590463876724, -1.0929937362670898],
    },
    "VAL": {
        "N": [0.9408224821090698, -1.2608877420425415, 0.652370810508728],
        "CA": [0.7287879586219788, -0.3937721848487854, -0.5120788216590881],
        "C": [1.7670493125915527, 0.6979415416717529, -0.5508439540863037],
        "O": [2.2449657917022705, 1.1683598756790161, 0.5171348452568054],
        "CB": [-0.7015937566757202, 0.21480272710323334, -0.5226467847824097],
        "CG1": [-1.7662651538848877, -0.845522403717041, -0.8411717414855957],
        "CG2": [-1.0500152111053467, 0.9374979138374329, 0.7892321944236755],
    },
    "UNK": {
        "N": [1.6134154796600342, -1.304404377937317, -0.35241976380348206],
        "CA": [0.6100443601608276, -0.23816797137260437, -0.3852081298828125],
        "C": [1.0861350297927856, 0.9232651591300964, 0.4423142373561859],
        "O": [1.374688982963562, 0.7629808187484741, 1.6591525077819824],
        "CB": [-0.7607048153877258, -0.7552146315574646, 0.10266945511102676],
    },
}


def _get_atom_position(res_name: str, atom_name: str, atom_idx: int) -> np.ndarray:
    """Get the canonical RDKit conformer position for an atom.

    Uses pre-extracted positions from official Boltz2 mol files. Falls back
    to a simple geometric placement for unknown residue/atom combinations.
    """
    if res_name in _RDKIT_CONFORMERS and atom_name in _RDKIT_CONFORMERS[res_name]:
        return np.array(_RDKIT_CONFORMERS[res_name][atom_name], dtype=np.float32)
    # Fallback for unknown atoms (should not happen for canonical AAs)
    angle = (atom_idx + 1) * 0.7
    radius = 1.4 + 0.03 * atom_idx
    return np.array(
        [
            radius * math.cos(angle),
            radius * math.sin(angle),
            0.1 * ((atom_idx % 5) - 2),
        ],
        dtype=np.float32,
    )


def _build_template(
    sequence: str,
) -> tuple[
    ProteinStructureTemplate,
    list[str],
    list[int],
    list[int],
    list[int],
    list[np.ndarray],
    list[int],
]:
    residue_names: list[str] = []
    residue_token_ids: list[int] = []
    atom_names: list[str] = []
    atom_elements: list[str] = []
    atom_residue_index: list[int] = []
    atom_chain_id: list[str] = []
    atom_positions: list[np.ndarray] = []
    residue_center_atom_idx: list[int] = []
    residue_disto_atom_idx: list[int] = []
    residue_frame_atom_idx: list[int] = []

    global_atom_idx = 0
    for res_idx, aa in enumerate(sequence):
        token_name = const.prot_letter_to_token[aa]
        residue_names.append(token_name)
        residue_token_ids.append(const.token_ids[token_name])

        residue_atoms = const.ref_atoms[token_name]
        if not residue_atoms:
            raise RuntimeError(f"No reference atoms for residue {token_name}.")
        center_atom_name = const.res_to_center_atom[token_name]
        disto_atom_name = const.res_to_disto_atom[token_name]

        center_idx = -1
        disto_idx = -1
        n_idx = -1
        ca_idx = -1
        c_idx = -1

        for local_idx, atom_name in enumerate(residue_atoms):
            atom_names.append(atom_name)
            element = _atom_name_to_element(atom_name)
            atom_elements.append(element)
            atom_residue_index.append(res_idx)
            atom_chain_id.append("A")

            atom_pos = _get_atom_position(token_name, atom_name, local_idx)
            atom_positions.append(atom_pos)

            if atom_name == center_atom_name:
                center_idx = global_atom_idx
            if atom_name == disto_atom_name:
                disto_idx = global_atom_idx
            if atom_name == "N":
                n_idx = global_atom_idx
            if atom_name == "CA":
                ca_idx = global_atom_idx
            if atom_name == "C":
                c_idx = global_atom_idx
            global_atom_idx += 1

        if center_idx == -1:
            center_idx = global_atom_idx - len(residue_atoms)
        if disto_idx == -1:
            disto_idx = center_idx
        if n_idx == -1:
            n_idx = center_idx
        if ca_idx == -1:
            ca_idx = center_idx
        if c_idx == -1:
            c_idx = center_idx

        residue_center_atom_idx.append(center_idx)
        residue_disto_atom_idx.append(disto_idx)
        residue_frame_atom_idx.extend([n_idx, ca_idx, c_idx])

    template = ProteinStructureTemplate(
        sequence=sequence,
        residue_names=residue_names,
        atom_names=atom_names,
        atom_elements=atom_elements,
        atom_residue_index=atom_residue_index,
        atom_chain_id=atom_chain_id,
    )

    return (
        template,
        residue_names,
        residue_token_ids,
        residue_center_atom_idx,
        residue_disto_atom_idx,
        atom_positions,
        residue_frame_atom_idx,
    )


def _random_rotation_matrix() -> torch.Tensor:
    """Sample a uniform random 3x3 rotation matrix (Algorithm 19 from AF2/Boltz)."""
    quaternion = torch.randn((1, 4), dtype=torch.float32)
    squared_norm = (quaternion * quaternion).sum(dim=1)
    norm = torch.sqrt(squared_norm)
    signed_norm = torch.where(quaternion[:, 0] < 0, -norm, norm)
    quaternion = quaternion / signed_norm[:, None]

    real, i, j, k = torch.unbind(quaternion, dim=-1)
    two_s = 2.0 / (quaternion * quaternion).sum(dim=-1)
    rotation = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * real),
            two_s * (i * k + j * real),
            two_s * (i * j + k * real),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * real),
            two_s * (i * k - j * real),
            two_s * (j * k + i * real),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return rotation.reshape(1, 3, 3)[0]


def _center_and_augment_atoms_per_residue(
    atom_positions: torch.Tensor,
    atom_residue_index: list[int],
    num_residues: int,
) -> torch.Tensor:
    """Apply Boltz2's seeded conformer augmentation per residue.

    The pinned implementation intentionally excludes the final residue because
    it iterates to the maximum reference-space identifier rather than through it.
    """
    result = atom_positions.clone()
    residue_index_tensor = torch.tensor(atom_residue_index, dtype=torch.long)
    for residue_idx in range(max(num_residues - 1, 0)):
        residue_mask = residue_index_tensor == residue_idx
        if not torch.any(residue_mask):
            raise RuntimeError(f"Residue index {residue_idx} has no atoms.")
        residue_coords = result[residue_mask][None]
        resolved_mask = torch.ones(
            residue_coords.shape[:2], dtype=torch.bool, device=residue_coords.device
        )
        residue_center = torch.sum(
            residue_coords * resolved_mask[:, :, None], dim=1, keepdim=True
        ) / torch.sum(resolved_mask[:, :, None], dim=1, keepdim=True)
        residue_coords = residue_coords - residue_center
        rotation = _random_rotation_matrix()[None]
        residue_coords = torch.einsum("bmd,bds->bms", residue_coords, rotation)
        residue_coords = residue_coords + torch.randn_like(residue_coords[:, 0:1, :])
        result[residue_mask] = residue_coords[0]
    return result


def build_boltz2_features(
    amino_acid_sequence: str,
    num_bins: int = 64,
    atoms_per_window_queries: int = 32,
) -> tuple[dict[str, torch.Tensor], ProteinStructureTemplate]:
    sequence = _normalize_sequence(amino_acid_sequence)
    (
        template,
        residue_names,
        residue_token_ids,
        residue_center_atom_idx,
        residue_disto_atom_idx,
        atom_positions_np,
        residue_frame_atom_idx_flat,
    ) = _build_template(sequence)

    num_tokens = len(residue_names)
    num_atoms = len(atom_positions_np)
    if num_tokens <= 0 or num_atoms <= 0:
        raise RuntimeError("Boltz2 feature construction produced an empty protein template.")

    atom_positions = torch.tensor(np.asarray(atom_positions_np), dtype=torch.float32)
    atom_positions = _center_and_augment_atoms_per_residue(
        atom_positions=atom_positions,
        atom_residue_index=template.atom_residue_index,
        num_residues=num_tokens,
    )

    token_index = torch.arange(num_tokens, dtype=torch.long).unsqueeze(0)
    residue_index = torch.arange(num_tokens, dtype=torch.long).unsqueeze(0)
    asym_id = torch.zeros((1, num_tokens), dtype=torch.long)
    entity_id = torch.zeros((1, num_tokens), dtype=torch.long)
    sym_id = torch.zeros((1, num_tokens), dtype=torch.long)
    mol_type = torch.full(
        (1, num_tokens),
        fill_value=const.chain_type_ids["PROTEIN"],
        dtype=torch.long,
    )

    res_type_ids = torch.tensor(residue_token_ids, dtype=torch.long)
    res_type = one_hot(res_type_ids, num_classes=const.num_tokens).unsqueeze(0)

    # token_bonds encodes explicit covalent cross-links from structure bonds,
    # NOT backbone peptide bonds (those are implicit via residue_index + asym_id).
    # For a standard single-chain protein without cross-links, this is all zeros.
    # This matches the official Boltz2 featurizer (featurizerv2.py lines 696-705).
    token_bonds = torch.zeros((num_tokens, num_tokens), dtype=torch.float32)
    type_bonds = torch.zeros((num_tokens, num_tokens), dtype=torch.long)
    token_bonds = token_bonds.unsqueeze(0).unsqueeze(-1)
    type_bonds = type_bonds.unsqueeze(0)

    token_pad_mask = torch.ones((1, num_tokens), dtype=torch.float32)
    token_resolved_mask = torch.ones((1, num_tokens), dtype=torch.float32)
    token_disto_mask = torch.ones((1, num_tokens), dtype=torch.float32)

    num_contact_classes = len(const.contact_conditioning_info)
    unspecified_id = const.contact_conditioning_info["UNSPECIFIED"]
    contact_ids = torch.full(
        (num_tokens, num_tokens),
        fill_value=unspecified_id,
        dtype=torch.long,
    )
    contact_conditioning = one_hot(
        contact_ids,
        num_classes=num_contact_classes,
    ).unsqueeze(0)
    contact_threshold = torch.zeros((1, num_tokens, num_tokens), dtype=torch.float32)

    if "x-ray diffraction" not in const.method_types_ids:
        raise RuntimeError("Boltz2 method metadata omits x-ray diffraction.")
    method_feature = torch.full(
        (1, num_tokens),
        fill_value=const.method_types_ids["x-ray diffraction"],
        dtype=torch.long,
    )
    modified = torch.zeros((1, num_tokens), dtype=torch.long)
    cyclic_period = torch.zeros((1, num_tokens), dtype=torch.float32)
    affinity_token_mask = torch.zeros((1, num_tokens), dtype=torch.float32)

    ref_pos = atom_positions.unsqueeze(0)
    atom_pad_mask = torch.ones((1, num_atoms), dtype=torch.float32)
    atom_resolved_mask = torch.ones((1, num_atoms), dtype=torch.bool)

    atom_name_codes = torch.stack(
        [_atom_name_to_codes(atom_name) for atom_name in template.atom_names],
        dim=0,
    )
    ref_atom_name_chars = one_hot(atom_name_codes, num_classes=64).unsqueeze(0)

    atomic_numbers = []
    for element in template.atom_elements:
        z_value = _ELEMENT_TO_Z[element] if element in _ELEMENT_TO_Z else _ELEMENT_TO_Z["C"]
        if z_value >= const.num_elements:
            raise RuntimeError(
                f"Atomic number {z_value} exceeds the Boltz2 element vocabulary."
            )
        atomic_numbers.append(z_value)
    ref_element = one_hot(
        torch.tensor(atomic_numbers, dtype=torch.long),
        num_classes=const.num_elements,
    ).unsqueeze(0)

    ref_charge_values = []
    ref_chirality_values = []
    for atom_name, residue_idx in zip(
        template.atom_names,
        template.atom_residue_index,
        strict=True,
    ):
        residue_name = residue_names[residue_idx]
        ref_charge_values.append(_FORMAL_CHARGES.get((residue_name, atom_name), 0.0))
        ref_chirality_values.append(
            2 if atom_name in _CHIRAL_ATOMS.get(residue_name, frozenset()) else 0
        )
    ref_charge = torch.tensor(ref_charge_values, dtype=torch.float32).unsqueeze(0)
    ref_chirality = torch.tensor(ref_chirality_values, dtype=torch.long).unsqueeze(0)
    ref_space_uid = torch.tensor(template.atom_residue_index, dtype=torch.long).unsqueeze(0)

    atom_to_token = one_hot(
        torch.tensor(template.atom_residue_index, dtype=torch.long),
        num_classes=num_tokens,
    ).unsqueeze(0)
    token_to_rep_atom = one_hot(
        torch.tensor(residue_disto_atom_idx, dtype=torch.long),
        num_classes=num_atoms,
    ).unsqueeze(0)
    token_to_center_atom = one_hot(
        torch.tensor(residue_center_atom_idx, dtype=torch.long),
        num_classes=num_atoms,
    ).unsqueeze(0)
    r_set_to_rep_atom = token_to_center_atom.clone()

    num_backbone_classes = (
        1 + len(const.protein_backbone_atom_index) + len(const.nucleic_backbone_atom_index)
    )
    backbone_ids = []
    for atom_name in template.atom_names:
        if atom_name in const.protein_backbone_atom_index:
            backbone_ids.append(const.protein_backbone_atom_index[atom_name] + 1)
        else:
            backbone_ids.append(0)
    atom_backbone_feat = one_hot(
        torch.tensor(backbone_ids, dtype=torch.long),
        num_classes=num_backbone_classes,
    ).unsqueeze(0)

    # X contains no observed coordinates for sequence-only inference.
    coords = torch.zeros((1, 1, num_atoms, 3), dtype=torch.float32)
    disto_coords_ensemble = torch.zeros(
        (1, 1, num_tokens, 3),
        dtype=torch.float32,
    )

    bfactor = torch.zeros((1, num_atoms), dtype=torch.float32)
    atom_plddt = torch.ones((1, num_atoms), dtype=torch.float32)

    if atoms_per_window_queries <= 0:
        raise ValueError("atoms_per_window_queries must be positive.")
    pad_atoms = (
        (num_atoms - 1) // atoms_per_window_queries + 1
    ) * atoms_per_window_queries - num_atoms
    if pad_atoms > 0:
        ref_pos = torch.nn.functional.pad(ref_pos, (0, 0, 0, pad_atoms), value=0.0)
        atom_pad_mask = torch.nn.functional.pad(atom_pad_mask, (0, pad_atoms), value=0.0)
        atom_resolved_mask = torch.nn.functional.pad(
            atom_resolved_mask,
            (0, pad_atoms),
            value=0.0,
        )
        ref_atom_name_chars = torch.nn.functional.pad(
            ref_atom_name_chars,
            (0, 0, 0, 0, 0, pad_atoms),
            value=0.0,
        )
        ref_element = torch.nn.functional.pad(ref_element, (0, 0, 0, pad_atoms), value=0.0)
        ref_charge = torch.nn.functional.pad(ref_charge, (0, pad_atoms), value=0.0)
        ref_chirality = torch.nn.functional.pad(ref_chirality, (0, pad_atoms), value=0)
        atom_backbone_feat = torch.nn.functional.pad(
            atom_backbone_feat,
            (0, 0, 0, pad_atoms),
            value=0.0,
        )
        ref_space_uid = torch.nn.functional.pad(ref_space_uid, (0, pad_atoms), value=0)
        coords = torch.nn.functional.pad(coords, (0, 0, 0, pad_atoms), value=0.0)
        atom_to_token = torch.nn.functional.pad(atom_to_token, (0, 0, 0, pad_atoms), value=0.0)
        token_to_rep_atom = torch.nn.functional.pad(
            token_to_rep_atom,
            (0, pad_atoms),
            value=0.0,
        )
        token_to_center_atom = torch.nn.functional.pad(
            token_to_center_atom,
            (0, pad_atoms),
            value=0.0,
        )
        r_set_to_rep_atom = torch.nn.functional.pad(
            r_set_to_rep_atom,
            (0, pad_atoms),
            value=0.0,
        )
        bfactor = torch.nn.functional.pad(bfactor, (0, pad_atoms), value=0.0)
        atom_plddt = torch.nn.functional.pad(atom_plddt, (0, pad_atoms), value=0.0)

    frames_idx = torch.tensor(
        residue_frame_atom_idx_flat,
        dtype=torch.long,
    ).reshape(num_tokens, 3)
    frames_idx = frames_idx.unsqueeze(0).unsqueeze(1)
    frame_resolved_mask = torch.zeros((1, 1, num_tokens), dtype=torch.bool)

    msa = torch.tensor(residue_token_ids, dtype=torch.long).unsqueeze(0).unsqueeze(0)
    msa_paired = torch.ones((1, 1, num_tokens), dtype=torch.float32)
    deletion_value = torch.zeros((1, 1, num_tokens), dtype=torch.float32)
    has_deletion = torch.zeros((1, 1, num_tokens), dtype=torch.bool)
    msa_mask = torch.ones((1, 1, num_tokens), dtype=torch.long)
    deletion_mean = torch.zeros((1, num_tokens), dtype=torch.float32)
    profile = (
        one_hot(
            torch.tensor(residue_token_ids, dtype=torch.long),
            num_classes=const.num_tokens,
        )
        .float()
        .unsqueeze(0)
    )

    template_restype = one_hot(
        torch.zeros((1, 1, num_tokens), dtype=torch.long),
        num_classes=const.num_tokens,
    )
    template_frame_rot = torch.zeros((1, 1, num_tokens, 3, 3), dtype=torch.float32)
    template_frame_t = torch.zeros((1, 1, num_tokens, 3), dtype=torch.float32)
    template_cb = torch.zeros((1, 1, num_tokens, 3), dtype=torch.float32)
    template_ca = torch.zeros((1, 1, num_tokens, 3), dtype=torch.float32)
    template_mask_cb = torch.zeros((1, 1, num_tokens), dtype=torch.float32)
    template_mask_frame = torch.zeros((1, 1, num_tokens), dtype=torch.float32)
    template_mask = torch.zeros((1, 1, num_tokens), dtype=torch.float32)
    query_to_template = torch.zeros((1, 1, num_tokens), dtype=torch.long)
    visibility_ids = torch.zeros((1, 1, num_tokens), dtype=torch.float32)

    disto_target = torch.zeros(
        (1, num_tokens, num_tokens, 1, num_bins),
        dtype=torch.float32,
    )
    disto_target[..., 0] = 1.0
    disto_center = torch.zeros((1, num_tokens, 3), dtype=torch.float32)

    features: dict[str, torch.Tensor] = {
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "res_type": res_type,
        "disto_center": disto_center,
        "token_bonds": token_bonds,
        "type_bonds": type_bonds,
        "token_pad_mask": token_pad_mask,
        "token_resolved_mask": token_resolved_mask,
        "token_disto_mask": token_disto_mask,
        "contact_conditioning": contact_conditioning,
        "contact_threshold": contact_threshold,
        "method_feature": method_feature,
        "modified": modified,
        "cyclic_period": cyclic_period,
        "affinity_token_mask": affinity_token_mask,
        "ref_pos": ref_pos,
        "atom_resolved_mask": atom_resolved_mask,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_chirality": ref_chirality,
        "atom_backbone_feat": atom_backbone_feat,
        "ref_space_uid": ref_space_uid,
        "coords": coords,
        "atom_pad_mask": atom_pad_mask,
        "atom_to_token": atom_to_token,
        "token_to_rep_atom": token_to_rep_atom,
        "r_set_to_rep_atom": r_set_to_rep_atom,
        "token_to_center_atom": token_to_center_atom,
        "disto_target": disto_target,
        "disto_coords_ensemble": disto_coords_ensemble,
        "bfactor": bfactor,
        "plddt": atom_plddt,
        "frames_idx": frames_idx,
        "frame_resolved_mask": frame_resolved_mask,
        "msa": msa,
        "msa_paired": msa_paired,
        "deletion_value": deletion_value,
        "has_deletion": has_deletion,
        "deletion_mean": deletion_mean,
        "profile": profile,
        "msa_mask": msa_mask,
        "template_restype": template_restype,
        "template_frame_rot": template_frame_rot,
        "template_frame_t": template_frame_t,
        "template_cb": template_cb,
        "template_ca": template_ca,
        "template_mask_cb": template_mask_cb,
        "template_mask_frame": template_mask_frame,
        "template_mask": template_mask,
        "query_to_template": query_to_template,
        "visibility_ids": visibility_ids,
    }

    return features, template
