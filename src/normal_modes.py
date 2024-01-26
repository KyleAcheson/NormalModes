import numpy as np
from pyscf import gto, dft
from pyscf.hessian.thermo import harmonic_analysis
import matplotlib.pyplot as plt

BOHR = 0.52917721092  # Angstroms
BOHR_SI = BOHR * 1e-10
ATOMIC_MASS = 1e-3 / 6.022140857e23
HARTREE2J = 4.359744650e-18
HARTREE2EV = 27.21138602
LIGHT_SPEED_SI = 299792458

AU2Hz = ((HARTREE2J / (ATOMIC_MASS * BOHR_SI ** 2)) ** 0.5 / (2 * np.pi))


def construct_mass_matrix(masses):
    mass_vec = np.repeat(masses, 3)
    mass_mat = np.sqrt(np.outer(mass_vec, mass_vec))
    return mass_mat


def diag_hessian(hessian):
    sq_mass_matrix = construct_mass_matrix(masses)
    weighted_hessian = hessian * (1 / sq_mass_matrix)
    fconstants, modes = np.linalg.eigh(weighted_hessian)
    freqs = np.sqrt(np.abs(fconstants)) # in a.u.
    freqs_wavenums = freqs * AU2Hz / LIGHT_SPEED_SI * 1e-2
    return freqs_wavenums, modes


def get_com(coordinates, masses):
    natoms = len(masses)
    tot = np.zeros((1, 3))
    for i in range(natoms):
        tot = tot + masses[i] * coordinates[i, :]
    centre_mass = tot / np.sum(masses)
    return centre_mass


def displace_normal_modes(ref_geom, normal_mode, dfactor, npoints):
    natoms = ref_geom.shape[0]
    displaced_coords = np.zeros((natoms, 3, npoints+1))
    #displaced_coords[:, :, 0] = ref_geom * (1 / BOHR)
    displaced_coords[:, :, 0] = ref_geom
    prev_geom = displaced_coords[:, :, 0]
    for i in range(npoints):
        new_geom = prev_geom + (dfactor * normal_mode)
        displaced_coords[:, :, i+1] = new_geom
        prev_geom = new_geom

    flipped_coords = np.flip(displaced_coords[:, :, :-1], axis=2)
    final_coords = np.concatenate((displaced_coords, flipped_coords), axis=2)

    return final_coords


def write_displacement(displaced_coords, atom_labels, fname):
    natoms, _, npoints = displaced_coords.shape
    with open(fname, 'a+') as f:
        for i in range(npoints):
            struct = displaced_coords[:, :, i] * BOHR
            f.write(f'{natoms}\n')
            f.write(f'displacement {i} \n')
            array_string = '\n'.join([f'{atom_labels[ind]} ' + ' '.join(map(str, row)) for ind, row in enumerate(struct)]) + '\n'
            f.write(array_string)


def normal_mode_analysis(hessian, masses, tol=1E-4):

    masses = np.array(masses)
    natoms = len(masses)
    nvib_modes = 3*natoms - 6

    # check hessian is symmetric
    diff = np.abs(hessian - hessian.T)
    is_symm = all(diff.flatten() < tol)
    if not is_symm:
        print(f'Warning: Hessian is NOT Symmetric (within {tol})?')

    freqs, modes = diag_hessian(hessian)
    vib_freqs = freqs[-nvib_modes:]
    print(f'Vibrational frequencies (cm^-1):\n{vib_freqs}')

    normal_modes = np.einsum('z,zri->izr', masses ** -.5, modes.reshape(natoms, 3, -1))

    return vib_freqs, normal_modes[-nvib_modes:, :, :], modes


def pyscf_nma(ref_geom, atom_labels):

    natoms = len(atom_labels)
    ref_geom = ref_geom.tolist()

    for i in range(natoms):
        ref_geom[i].insert(0, atom_labels[i])

    mol = gto.M(atom=ref_geom, basis='def2-svp', verbose=0)
    mf = dft.RKS(mol)
    mf.xc = 'B3LYP'
    mf.kernel()

    hessian = mf.Hessian().kernel()
    freqs = harmonic_analysis(mol, hessian)['freq_wavenumber']
    norm_modes = harmonic_analysis(mol, hessian)['norm_mode']
    hessian = hessian.transpose(0, 2, 1, 3).reshape(natoms * 3, natoms * 3)
    return freqs, hessian, norm_modes


def normal_coordinate_transform(coords, tmat):
    return tmat.T @ coords.flatten()

def mass_weighted_displacements(coords, ref, masses):
    natoms, _, npoints = coords.shape
    mw_dcoords = np.zeros((natoms, 3, npoints))
    #ref *= (1 / BOHR)
    for i in range(npoints):
        mw_dcoords[:, :, i] = (coords[:, :, i] - ref) * np.sqrt(masses[:, np.newaxis])

    return mw_dcoords


def transform_trajectory(coordinates, tmat, ref, masses):
    mw_disp_coords = mass_weighted_displacements(coordinates, ref, masses)
    natoms, _, npoints = coordinates.shape
    q_coords = np.zeros((3*natoms, npoints))
    for i in range(npoints):
        q = normal_coordinate_transform(mw_disp_coords[:, :, i], tmat)
        q_coords[:, i] = q
    return q_coords


def inverse_transform_trajectory(q_coords, tmat, ref_geom, masses):
    nmodes, npoints = q_coords.shape
    natoms = len(masses)
    cartesian_coords = np.zeros((natoms, 3, npoints))
    for i in range(npoints):
        c = tmat @ q_coords[:, i]
        cartesian_coords[:, :, i] = c.reshape((natoms, 3)) * (1 / np.sqrt(masses[:, np.newaxis]))
        cartesian_coords[:, :, i] += ref_geom
    return cartesian_coords


if __name__ == "__main__":
    #hessian_file = 'H2O_pes/hessian.txt'
    mode = 0
    npoints = 10
    atom_labels = ['O', 'H', 'H']
    masses = np.array([15.999, 1.008, 1.008])
    natoms = len(masses)

    #hessian = np.genfromtxt(hessian_file)
    #vib_freqs, normal_modes, trans_matrix = normal_mode_analysis(hessian, masses)

    ref_geom = np.genfromtxt('H2O_pes/freq/test_orca.xyz', delimiter=', ').reshape(natoms, 3)
    py_vib_freqs, hessian, py_normal_modes = pyscf_nma(ref_geom, atom_labels)
    ref_geom *= (1 / BOHR)

    print('PYSCF Vibrational Frequencies (cm^-1):')
    print(py_vib_freqs)

    vib_freqs, normal_modes, eigvecs = normal_mode_analysis(hessian, masses)

    nmode_diff = np.abs(normal_modes - py_normal_modes)

    displaced_coords = displace_normal_modes(ref_geom, normal_modes[mode, :, :], 0.1, npoints)
    npoints = displaced_coords.shape[-1]
    #write_displacement(displaced_coords, atom_labels, fname=f'displacement_mode{mode}.txt')
    q_coords = transform_trajectory(displaced_coords, eigvecs, ref_geom, masses)


    qvib_coords = q_coords[-3:, :]
    vib_labels = ['bend', 'sym', 'aym']
    plt.figure()
    for i in range(3):
        plt.plot(qvib_coords[i, :], label=f'{vib_labels[i]}')
    plt.legend()
    plt.ylabel('Q')
    plt.savefig(f'displacement{mode}_nmcoords.png')

    reversed_cartesians = inverse_transform_trajectory(q_coords, eigvecs, ref_geom, masses)
    write_displacement(reversed_cartesians, atom_labels, f'inverse_transform_mode{mode}.xyz')

    breakpoint()