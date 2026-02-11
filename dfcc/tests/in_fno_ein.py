"""
Test DF-CCSD(T) equation solution
Psi4: fnocc
Ref: J. Chem. Theory Comput. 2013, 9, 6, 2687–2696
    https://doi.org/10.1021/ct400250u
"""

import sys
print("Python executable:", sys.executable)
print("PYTHONPATH:", sys.path)

import psi4
from dfcc.fnocc import ccwfn

def df_h2o():
    # Psi4 Setup
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pVDZ',
                      'df_basis_scf': 'cc-pvdz-jkfit',
                      #'df_basis_cc': 'cc-pvdz-ri',
                      'scf_type': 'df',
                      'freeze_core': 'true',
                      'guess':'gwh',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,})

    mol = psi4.geometry(
        """
        O
        H 1 1.1
        H 1 1.1 2 104
        symmetry c1
        """)
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    # Two auxiliary sets are needed for DF-CC
    # T1-transformation
    aux_bas_scf = psi4.core.BasisSet.build(mol, 'DF_BASIS_MP2', '', 'JKFIT', psi4.core.get_global_option('BASIS'))
    # Residuals
    aux_bas_cc = psi4.core.BasisSet.build(mol, 'DF_BASIS_MP2', '', 'RIFIT', psi4.core.get_global_option('BASIS'))

    return rhf_e, rhf_wfn, aux_bas_scf, aux_bas_cc

eref, ref, aux_bas_scf, aux_bas_cc = df_h2o()

maxiter = 75
e_conv = 1e-12
r_conv = 1e-12

cc = ccwfn(ref, aux_bas_scf, aux_bas_cc, model='DF-CCSD(T)', variant='ein')

ecc = cc.solve_cc(e_conv, r_conv, maxiter)
eccsd_ref = -0.222144798751
#t_ref = -0.00386579686378
print("------------------------------")
print("eccsd_ref: ", eccsd_ref)
print("ecc: ", ecc)
#print("(t)_ref: ", t_ref)
#print("eccsd(t)_ref: ", eccsd_ref + t_ref)
#print("eccsd(t):", ecc)
print((ecc - eccsd_ref) < 1E-12)

# Eugene (fnocc):
# eccsd_ref = -0.222144798751
# eccsd(t)_ref = -0.226010593271

# Uger (dfnocc):
# eccsd_ref = -0.22214471546645
