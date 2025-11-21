"""
Physics Engine for Feynman Diagram Generation
Mirrors the constants and validation logic from feynman-logic.js
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np

@dataclass
class Particle:
    """Base particle class with quantum numbers"""
    id: str
    name: str
    en_name: str
    symbol: str
    charge: float  # Electric charge
    lepton: float  # Lepton number
    baryon: float  # Baryon number
    flavor: str
    spin: float
    parity: int  # +1 or -1
    mass: float  # in MeV
    color: Optional[str] = None  # For quarks: 'red', 'green', 'blue', 'anti-red', etc.
    
    def __post_init__(self):
        """Validate particle properties"""
        assert self.parity in [-1, 1], f"Invalid parity: {self.parity}"
        assert self.spin >= 0, f"Invalid spin: {self.spin}"
        assert self.mass >= 0, f"Invalid mass: {self.mass}"

@dataclass
class Boson:
    """Boson particle class"""
    id: str
    name: str
    en_name: str
    symbol: str
    charge: float
    lepton: float
    baryon: float
    spin: float
    parity: int
    mass: float
    flavor_change: bool = False  # True for W bosons
    color_charge: Optional[Tuple[str, str]] = None  # For gluons: (color_in, color_out)
    
    def __post_init__(self):
        assert self.parity in [-1, 1]
        assert self.spin >= 0


class PhysicsConstants:
    """
    Standard Model constants mirroring feynman-logic.js PHYSICS object
    """
    
    # Leptons
    LEPTONS = [
        Particle('e', '电子 (e⁻)', 'Electron (e⁻)', 'e⁻', -1, 1, 0, 'electron', 0.5, 1, 0.511),
        Particle('mu', '缪子 (μ⁻)', 'Muon (μ⁻)', 'μ⁻', -1, 1, 0, 'muon', 0.5, 1, 105.7),
        Particle('tau', '陶子 (τ⁻)', 'Tau (τ⁻)', 'τ⁻', -1, 1, 0, 'tau', 0.5, 1, 1777),
        Particle('nu_e', '电中微子 (νₑ)', 'Electron neutrino (νₑ)', 'νₑ', 0, 1, 0, 'electron', 0.5, -1, 0.001),
        Particle('nu_mu', '缪中微子 (ν_μ)', 'Muon neutrino (ν_μ)', 'ν_μ', 0, 1, 0, 'muon', 0.5, -1, 0.001),
        Particle('nu_tau', '陶中微子 (ν_τ)', 'Tau neutrino (ν_τ)', 'ν_τ', 0, 1, 0, 'tau', 0.5, -1, 0.001),
    ]
    
    # Up-type Quarks
    QUARKS_U = [
        Particle('u', '上夸克 (u)', 'Up quark (u)', 'u', 2/3, 0, 1/3, 'up', 0.5, 1, 2.2),
        Particle('c', '魅力 (c)', 'Charm (c)', 'c', 2/3, 0, 1/3, 'charm', 0.5, 1, 1280),
        Particle('t', '顶夸克 (t)', 'Top (t)', 't', 2/3, 0, 1/3, 'top', 0.5, 1, 173000),
    ]
    
    # Down-type Quarks
    QUARKS_D = [
        Particle('d', '下夸克 (d)', 'Down quark (d)', 'd', -1/3, 0, 1/3, 'down', 0.5, 1, 4.7),
        Particle('s', '奇异 (s)', 'Strange (s)', 's', -1/3, 0, 1/3, 'strange', 0.5, 1, 96),
        Particle('b', '底夸克 (b)', 'Bottom (b)', 'b', -1/3, 0, 1/3, 'bottom', 0.5, 1, 4180),
    ]
    
    # Bosons
    BOSONS = {
        'photon': Boson('photon', '光子 (γ)', 'Photon (γ)', 'γ', 0, 0, 0, 1, -1, 0.0),
        'gluon': Boson('gluon', '胶子 (g)', 'Gluon (g)', 'g', 0, 0, 0, 1, -1, 0.0),
        'w_plus': Boson('w_plus', 'W⁺', 'W⁺', 'W⁺', 1, 0, 0, 1, 1, 80379, flavor_change=True),
        'w_minus': Boson('w_minus', 'W⁻', 'W⁻', 'W⁻', -1, 0, 0, 1, 1, 80379, flavor_change=True),
        'z': Boson('z', 'Z⁰', 'Z⁰', 'Z⁰', 0, 0, 0, 1, 1, 91188),
        'higgs': Boson('higgs', '希格斯 (H)', 'Higgs (H)', 'H', 0, 0, 0, 0, 1, 125100),
    }
    
    # Color charges for quarks
    COLORS = ['red', 'green', 'blue', 'anti-red', 'anti-green', 'anti-blue']
    
    # Gluon color combinations (color-anticolor pairs)
    GLUON_COLORS = [
        ('red', 'anti-green'),
        ('red', 'anti-blue'),
        ('green', 'anti-red'),
        ('green', 'anti-blue'),
        ('blue', 'anti-red'),
        ('blue', 'anti-green'),
    ]
    
    # CKM Matrix for quark flavor mixing (Weak interactions)
    CKM_MATRIX = {
        'u': {'d': 0.97370, 's': 0.2245, 'b': 0.00382},
        'c': {'d': 0.221, 's': 0.987, 'b': 0.041},
        't': {'d': 0.008, 's': 0.0388, 'b': 1.013}
    }
    
    # Coupling constants
    ALPHA_EM = 1.0 / 137.0      # Electromagnetic
    ALPHA_S = 0.1181            # Strong (QCD)
    ALPHA_W = 1.0 / 30.0        # Weak
    
    # Dimensionality constants
    FERMION_DIMENSIONALITY = 1.5
    BOSON_DIMENSIONALITY = 1.0
    MAXIMUM_DIMENSIONALITY = 4.0
    
    @classmethod
    def get_all_particles(cls) -> List:
        """Get all fermions (leptons + quarks)"""
        return cls.LEPTONS + cls.QUARKS_U + cls.QUARKS_D
    
    @classmethod
    def get_particle_by_id(cls, particle_id: str) -> Optional[Particle]:
        """Retrieve particle by ID"""
        for p in cls.get_all_particles():
            if p.id == particle_id:
                return p
        return None
    
    @classmethod
    def get_boson_by_id(cls, boson_id: str) -> Optional[Boson]:
        """Retrieve boson by ID"""
        return cls.BOSONS.get(boson_id)
    
    @classmethod
    def is_quark(cls, particle_id: str) -> bool:
        """Check if particle is a quark"""
        for q in cls.QUARKS_U + cls.QUARKS_D:
            if q.id == particle_id:
                return True
        return False
    
    @classmethod
    def is_lepton(cls, particle_id: str) -> bool:
        """Check if particle is a lepton"""
        for l in cls.LEPTONS:
            if l.id == particle_id:
                return True
        return False
    
    @classmethod
    def has_charge(cls, particle_id: str) -> bool:
        """Check if particle has electric charge"""
        p = cls.get_particle_by_id(particle_id)
        if p:
            return abs(p.charge) > 1e-6
        b = cls.get_boson_by_id(particle_id)
        if b:
            return abs(b.charge) > 1e-6
        return False


class ConservationLaws:
    """
    Physics validation logic implementing Kirchhoff's laws for quantum numbers
    """
    
    @staticmethod
    def check_charge_conservation(incoming: List[float], outgoing: List[float], tolerance=1e-6) -> Tuple[bool, float]:
        """
        Check charge conservation: ΣQ_in = ΣQ_out
        Returns: (is_conserved, mismatch)
        """
        q_in = sum(incoming)
        q_out = sum(outgoing)
        mismatch = abs(q_in - q_out)
        return mismatch < tolerance, mismatch
    
    @staticmethod
    def check_lepton_conservation(incoming: List[float], outgoing: List[float], tolerance=1e-6) -> Tuple[bool, float]:
        """Check lepton number conservation"""
        l_in = sum(incoming)
        l_out = sum(outgoing)
        mismatch = abs(l_in - l_out)
        return mismatch < tolerance, mismatch
    
    @staticmethod
    def check_baryon_conservation(incoming: List[float], outgoing: List[float], tolerance=1e-6) -> Tuple[bool, float]:
        """Check baryon number conservation"""
        b_in = sum(incoming)
        b_out = sum(outgoing)
        mismatch = abs(b_in - b_out)
        return mismatch < tolerance, mismatch
    
    @staticmethod
    def check_color_conservation(incoming_colors: List[Optional[str]], 
                                 outgoing_colors: List[Optional[str]]) -> Tuple[bool, float]:
        """
        Check color charge conservation (QCD)
        Colors must form a color singlet (white)
        Returns: (is_conserved, mismatch_score)
        """
        # Collect all colors
        all_colors = []
        for c in incoming_colors:
            if c:
                all_colors.append(c)
        for c in outgoing_colors:
            if c:
                # Outgoing reverses the color charge
                if c.startswith('anti-'):
                    all_colors.append(c[5:])  # anti-red -> red
                else:
                    all_colors.append('anti-' + c)  # red -> anti-red
        
        # Count colors
        color_count = {'red': 0, 'green': 0, 'blue': 0}
        for c in all_colors:
            if c.startswith('anti-'):
                color_count[c[5:]] -= 1
            else:
                color_count[c] += 1
        
        # Check if all colors cancel out (color singlet)
        mismatch = sum(abs(v) for v in color_count.values())
        return mismatch == 0, float(mismatch)
    
    @staticmethod
    def check_spin_conservation(incoming_spins: List[float], 
                               outgoing_spins: List[float]) -> Tuple[bool, float]:
        """
        Check angular momentum conservation
        Note: This is a simplified check. Full spin coupling is more complex.
        """
        total_in = sum(incoming_spins)
        total_out = sum(outgoing_spins)
        # Allow for orbital angular momentum
        mismatch = abs(total_in - total_out)
        # Spins must differ by integer values
        is_valid = abs(mismatch - round(mismatch)) < 0.1
        return is_valid, mismatch
    
    @staticmethod
    def check_vertex_dimensionality(particle_ids: List[str]) -> Tuple[bool, float]:
        """
        Check that vertex dimensionality does not exceed 4.0
        Prevents non-renormalizable interactions
        """
        total_dim = 0.0
        for pid in particle_ids:
            p = PhysicsConstants.get_particle_by_id(pid)
            if p:
                total_dim += PhysicsConstants.FERMION_DIMENSIONALITY
            else:
                total_dim += PhysicsConstants.BOSON_DIMENSIONALITY
        
        is_valid = total_dim <= PhysicsConstants.MAXIMUM_DIMENSIONALITY
        return is_valid, total_dim
    
    @staticmethod
    def check_interaction_rules(vertex_particles: List[str]) -> Tuple[bool, List[str]]:
        """
        Check Standard Model interaction rules:
        - Photon only couples to charged particles
        - Gluon only couples to quarks
        - Higgs only couples to massive particles
        Returns: (is_valid, list of violations)
        """
        violations = []
        
        # Check if photon is present
        if 'photon' in vertex_particles:
            other_particles = [p for p in vertex_particles if p != 'photon']
            for p_id in other_particles:
                if not PhysicsConstants.has_charge(p_id):
                    violations.append(f"Photon coupled to uncharged particle {p_id}")
        
        # Check if gluon is present
        if 'gluon' in vertex_particles:
            other_particles = [p for p in vertex_particles if p != 'gluon']
            for p_id in other_particles:
                if not PhysicsConstants.is_quark(p_id) and p_id != 'gluon':
                    violations.append(f"Gluon coupled to non-quark particle {p_id}")
        
        # Check if Higgs is present
        if 'higgs' in vertex_particles:
            other_particles = [p for p in vertex_particles if p != 'higgs']
            for p_id in other_particles:
                p = PhysicsConstants.get_particle_by_id(p_id)
                b = PhysicsConstants.get_boson_by_id(p_id)
                mass = p.mass if p else (b.mass if b else 0)
                if mass < 1e-3:  # Massless
                    violations.append(f"Higgs coupled to massless particle {p_id}")
        
        return len(violations) == 0, violations


class AntiparticleHelper:
    """Helper functions for handling antiparticles"""
    
    @staticmethod
    def get_antiparticle_properties(particle: Particle) -> Dict:
        """
        Get properties of the antiparticle
        Charge, Lepton, Baryon numbers flip sign
        Color charge becomes anti-color
        """
        anti_color = None
        if particle.color:
            if particle.color.startswith('anti-'):
                anti_color = particle.color[5:]
            else:
                anti_color = 'anti-' + particle.color
        
        return {
            'id': particle.id,
            'symbol': particle.symbol,
            'charge': -particle.charge,
            'lepton': -particle.lepton,
            'baryon': -particle.baryon,
            'spin': particle.spin,
            'parity': particle.parity,
            'mass': particle.mass,
            'color': anti_color,
            'is_anti': True
        }
    
    @staticmethod
    def convert_to_anti_symbol(symbol: str) -> str:
        """Convert particle symbol to antiparticle symbol"""
        if symbol.endswith('⁻'):
            return symbol[:-1] + '⁺'
        elif symbol.endswith('⁺'):
            return symbol[:-1] + '⁻'
        else:
            return symbol + '\u0305'  # Add overline for anti


# Precompute particle feature encodings for neural network
class ParticleEncoder:
    """Encode particles as fixed-size feature vectors for neural networks"""
    
    # Particle type encoding (one-hot style)
    TYPE_ENCODING = {
        'lepton': 0,
        'quark_u': 1,
        'quark_d': 2,
        'photon': 3,
        'gluon': 4,
        'w_plus': 5,
        'w_minus': 6,
        'z': 7,
        'higgs': 8
    }
    
    @staticmethod
    def encode_particle(particle_id: str, is_anti: bool = False, color: Optional[str] = None) -> np.ndarray:
        """
        Encode a particle as a feature vector:
        [Type(9), Spin(1), Charge(1), Lepton(1), Baryon(1), Color(6), IsAnti(1), Mass(1)]
        Total: 21 features
        """
        p = PhysicsConstants.get_particle_by_id(particle_id)
        b = PhysicsConstants.get_boson_by_id(particle_id)
        
        if p is None and b is None:
            raise ValueError(f"Unknown particle ID: {particle_id}")
        
        # Type encoding (one-hot)
        type_vec = np.zeros(9)
        if p:
            if PhysicsConstants.is_lepton(particle_id):
                type_vec[0] = 1
            elif particle_id in ['u', 'c', 't']:
                type_vec[1] = 1
            elif particle_id in ['d', 's', 'b']:
                type_vec[2] = 1
        elif b:
            type_vec[ParticleEncoder.TYPE_ENCODING[particle_id]] = 1
        
        # Quantum numbers
        if p:
            spin = p.spin
            charge = -p.charge if is_anti else p.charge
            lepton = -p.lepton if is_anti else p.lepton
            baryon = -p.baryon if is_anti else p.baryon
            mass = p.mass
        else:
            spin = b.spin
            charge = b.charge
            lepton = b.lepton
            baryon = b.baryon
            mass = b.mass
        
        # Color encoding (one-hot for 6 colors)
        color_vec = np.zeros(6)
        if color:
            color_idx = PhysicsConstants.COLORS.index(color)
            color_vec[color_idx] = 1
        
        # IsAnti flag
        is_anti_flag = 1.0 if is_anti else 0.0
        
        # Normalize mass (log scale)
        mass_normalized = np.log10(mass + 1e-3)
        
        return np.concatenate([
            type_vec,           # 9 features
            [spin],             # 1 feature
            [charge],           # 1 feature
            [lepton],           # 1 feature
            [baryon],           # 1 feature
            color_vec,          # 6 features
            [is_anti_flag],     # 1 feature
            [mass_normalized]   # 1 feature
        ])  # Total: 21 features
