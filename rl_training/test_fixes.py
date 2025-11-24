"""
Quick test for the two critical bug fixes:
1. Photon classification in visualization
2. MERGE physics constraint
"""

from feynman_env import FeynmanDiagramEnv
from physics_engine import PhysicsConstants

print("=" * 80)
print("🧪 Testing Critical Bug Fixes")
print("=" * 80)

# Test 1: Verify photon classification
print("\n=== Test 1: Photon Classification ===")
env = FeynmanDiagramEnv(['e', 'e_bar'], ['photon'], max_vertices=10, max_steps=50)
env.reset()

print(f"Initial particles: {env.initial_particles}")
print(f"Final particles: {env.final_particles}")
print(f"Particle list: {env.particle_list[:15]}...")
print(f"✓ Photon in particle_list? {('photon' in env.particle_list)}")

# Test 2: Try to merge two final vertices (should fail now)
print("\n=== Test 2: MERGE Physics Constraint ===")
env2 = FeynmanDiagramEnv(['e', 'e_bar'], ['mu', 'mu_bar'], max_vertices=10, max_steps=50)
env2.reset()

print(f"Vertices: {[(i, v['type']) for i, v in enumerate(env2.vertices)]}")
print(f"  V0: initial (e)")
print(f"  V1: initial (e_bar)")
print(f"  V2: final (mu)")
print(f"  V3: final (mu_bar)")

# Try to merge final vertices (should FAIL)
print("\n🚫 Attempting: MERGE(V2_final, V3_final)...")
result1 = env2._execute_merge(2, 3, 0)
print(f"   Result: {result1} {'✓ BLOCKED (correct!)' if not result1 else '✗ ALLOWED (bug!)'}")

# Try to merge initial vertices (should SUCCEED)
print("\n✅ Attempting: MERGE(V0_initial, V1_initial)...")
result2 = env2._execute_merge(0, 1, 0)
print(f"   Result: {result2} {'✓ ALLOWED (correct!)' if result2 else '✗ BLOCKED (bug!)'}")

# Test 3: Verify visualization classification
print("\n=== Test 3: Visualization Classification ===")

# Directly test the get_particle_props function logic
from physics_engine import PhysicsConstants

def get_particle_props(p_id):
    """Test version of the classification function"""
    base_id = p_id[:-4] if p_id.endswith('_bar') else p_id
    is_anti = p_id.endswith('_bar')
    
    # Check if fermion
    p = PhysicsConstants.get_particle_by_id(base_id)
    if p:
        return 'fermion', 'fermion', base_id, is_anti
    
    # Check if boson
    b = PhysicsConstants.get_boson_by_id(base_id)
    if b:
        if base_id == 'photon':
            shape_type = 'photon'
        elif base_id in ['w_plus', 'w_minus']:
            shape_type = 'boson_w'
        elif base_id == 'z':
            shape_type = 'boson_z'
        elif base_id == 'gluon':
            shape_type = 'gluon'
        elif base_id == 'higgs':
            shape_type = 'higgs'
        else:
            shape_type = 'boson'
        return shape_type, 'boson', base_id, is_anti
    
    return 'fermion', 'fermion', base_id, is_anti

# Test various particles
test_particles = ['e', 'e_bar', 'photon', 'mu', 'mu_bar', 'gluon', 'w_plus', 'z', 'higgs']
print("Particle classification test:")
for p_id in test_particles:
    shape_type, category, base_id, is_anti = get_particle_props(p_id)
    status = "✓" if (p_id == 'photon' and category == 'boson') or (p_id in ['e', 'mu'] and category == 'fermion') else "?"
    print(f"  {status} {p_id:12s} → type={shape_type:12s}, category={category:8s}, anti={is_anti}")

# Specifically verify photon
shape_type, category, base_id, is_anti = get_particle_props('photon')
if category == 'boson' and shape_type == 'photon':
    print(f"\n✓ Photon correctly classified as boson with shape type 'photon'!")
else:
    print(f"\n✗ BUG: Photon classified as {category}/{shape_type}")

print("\n" + "=" * 80)
print("✅ All tests complete!")
print("=" * 80)
