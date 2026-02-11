"""
Test DF-CCSD(T) equation solution
Psi4: dfocc
Ref: notes_dfocc.pdf
"""

import sys
print("Python executable:", sys.executable)
print("PYTHONPATH:", sys.path)

import psi4
from dfcc.dfocc import ccwfn

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
    aux_bas = psi4.core.BasisSet.build(mol, 'DF_BASIS_MP2', '', 'RIFIT', psi4.core.get_global_option('BASIS'))

    return rhf_e, rhf_wfn, aux_bas

eref, ref, aux_bas = df_h2o()

maxiter = 75
e_conv = 1e-12
r_conv = 1e-12

cc = ccwfn(ref, aux_bas, model='DF-CCSD(T)', variant='ein')

ecc = cc.solve_cc(e_conv, r_conv, maxiter)

eccsd_ref = -0.22214471567819
t_ref = -0.00386579686378
print("------------------------------")
print("eccsd_ref: ", eccsd_ref)
print("(t)_ref: ", t_ref)
print("eccsd(t)_ref: ", eccsd_ref + t_ref)
print("eccsd(t):", ecc)
print(abs(eccsd_ref + t_ref - ecc) < 1e-7)

## Ref energies when conv = 1e-12

# Eugene:
# eccsd_ref = -0.222144798751
# (t)_ref = -0.003865794519

# Uger:
# eccsd_ref = -0.22214471546645
# (t)_ref = - 0.00386579685119

# 01/14
# -0.222144730056436
# -0.222144730056436

# -0.426638869699597
#-0.426638869699594
