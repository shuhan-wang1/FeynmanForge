"""
Particle Database and Utility Functions
Helper functions for working with particles in Feynman diagram generation
"""

from typing import List, Tuple, Optional
from physics_engine import PhysicsConstants, Particle, Boson
import numpy as np


def get_particle_list() -> List[str]:
    """Get list of all particle IDs (fermions + bosons)"""
    all_particles = [p.id for p in PhysicsConstants.get_all_particles()]
    all_bosons = list(PhysicsConstants.BOSONS.keys())
    return all_particles + all_bosons


def get_particle_index(particle_id: str) -> int:
    """Get index of particle in the particle list"""
    particle_list = get_particle_list()
    return particle_list.index(particle_id)


def parse_particle_string(particle_str: str) -> Tuple[str, bool]:
    """
    Parse particle string with antiparticle notation

    Examples:
        'e' -> ('e', False)
        'e_bar' -> ('e', True)
        'mu_bar' -> ('mu', True)

    Returns:
        (particle_id, is_anti)
    """
    if particle_str.endswith('_bar'):
        return particle_str.replace('_bar', ''), True
    return particle_str, False


def get_quantum_numbers(particle_id: str, is_anti: bool = False) -> dict:
    """
    Get quantum numbers for a particle

    Returns:
        Dictionary with charge, lepton, baryon, spin, mass
    """
    p = PhysicsConstants.get_particle_by_id(particle_id)
    b = PhysicsConstants.get_boson_by_id(particle_id)

    if p:
        return {
            'charge': -p.charge if is_anti else p.charge,
            'lepton': -p.lepton if is_anti else p.lepton,
            'baryon': -p.baryon if is_anti else p.baryon,
            'spin': p.spin,
            'mass': p.mass,
            'flavor': p.flavor,
            'is_fermion': True
        }
    elif b:
        return {
            'charge': b.charge,
            'lepton': b.lepton,
            'baryon': b.baryon,
            'spin': b.spin,
            'mass': b.mass,
            'flavor': None,
            'is_fermion': False
        }
    else:
        raise ValueError(f"Unknown particle: {particle_id}")


def get_particle_symbol(particle_id: str, is_anti: bool = False) -> str:
    """Get LaTeX/Unicode symbol for particle"""
    p = PhysicsConstants.get_particle_by_id(particle_id)
    b = PhysicsConstants.get_boson_by_id(particle_id)

    if p:
        symbol = p.symbol
        if is_anti:
            # Convert to antiparticle symbol
            if symbol.endswith('⁻'):
                return symbol[:-1] + '⁺'
            elif symbol.endswith('⁺'):
                return symbol[:-1] + '⁻'
            else:
                return symbol + '\u0305'  # Add overline
        return symbol
    elif b:
        return b.symbol
    else:
        return particle_id


def validate_reaction(initial_particles: List[str], final_particles: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate if a reaction is theoretically possible

    Checks:
    1. All particles exist
    2. Conservation laws are satisfied (Q, L, B)

    Returns:
        (is_valid, list of error messages)
    """
    errors = []

    # Check all particles exist
    particle_list = get_particle_list()
    for p_str in initial_particles + final_particles:
        p_id, is_anti = parse_particle_string(p_str)
        if p_id not in particle_list:
            errors.append(f"Unknown particle: {p_id}")

    if errors:
        return False, errors

    # Check conservation laws
    q_in = []
    l_in = []
    b_in = []
    for p_str in initial_particles:
        p_id, is_anti = parse_particle_string(p_str)
        qn = get_quantum_numbers(p_id, is_anti)
        q_in.append(qn['charge'])
        l_in.append(qn['lepton'])
        b_in.append(qn['baryon'])

    q_out = []
    l_out = []
    b_out = []
    for p_str in final_particles:
        p_id, is_anti = parse_particle_string(p_str)
        qn = get_quantum_numbers(p_id, is_anti)
        q_out.append(qn['charge'])
        l_out.append(qn['lepton'])
        b_out.append(qn['baryon'])

    # Check conservation
    q_total_in = sum(q_in)
    q_total_out = sum(q_out)
    if abs(q_total_in - q_total_out) > 1e-6:
        errors.append(f"Charge not conserved: {q_total_in:.2f} -> {q_total_out:.2f}")

    l_total_in = sum(l_in)
    l_total_out = sum(l_out)
    if abs(l_total_in - l_total_out) > 1e-6:
        errors.append(f"Lepton number not conserved: {l_total_in:.2f} -> {l_total_out:.2f}")

    b_total_in = sum(b_in)
    b_total_out = sum(b_out)
    if abs(b_total_in - b_total_out) > 1e-6:
        errors.append(f"Baryon number not conserved: {b_total_in:.2f} -> {b_total_out:.2f}")

    return len(errors) == 0, errors


def get_reaction_string(initial_particles: List[str], final_particles: List[str]) -> str:
    """
    Format reaction as string with symbols

    Example:
        ['e', 'e_bar'] -> ['mu', 'mu_bar']
        Returns: "e⁻ + e⁺ → μ⁻ + μ⁺"
    """
    initial_symbols = [get_particle_symbol(*parse_particle_string(p)) for p in initial_particles]
    final_symbols = [get_particle_symbol(*parse_particle_string(p)) for p in final_particles]

    initial_str = ' + '.join(initial_symbols)
    final_str = ' + '.join(final_symbols)

    return f"{initial_str} → {final_str}"


def categorize_particles_by_type() -> dict:
    """Categorize all particles by type (leptons, quarks, bosons)"""
    return {
        'leptons': [p.id for p in PhysicsConstants.LEPTONS],
        'quarks_up': [p.id for p in PhysicsConstants.QUARKS_U],
        'quarks_down': [p.id for p in PhysicsConstants.QUARKS_D],
        'bosons': list(PhysicsConstants.BOSONS.keys())
    }


def get_baryon_number_particles() -> List[str]:
    """Get list of particles with non-zero baryon number (quarks)"""
    particles_with_baryon = []
    for p in PhysicsConstants.get_all_particles():
        if abs(p.baryon) > 1e-6:
            particles_with_baryon.append(p.id)
    return particles_with_baryon
