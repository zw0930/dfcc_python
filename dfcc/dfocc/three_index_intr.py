"""
three_ind_intr.py: creates 3-index intermediates for T1, T2 residuals
"""

# tijQ = sum_e (t1_i^e B_je^Q)
def build_tijQ(t1, BovQ, contract):
    return contract("ie,jeQ->ijQ", t1, BovQ)

# tiaQ = sum_f (t1_i^f B_fa^Q)
def build_tiaQ(t1, BvvQ, contract):
    return contract("if,faQ->iaQ", t1, BvvQ)

# taiQ = sum_m (t1_m^a B_mi^Q)
def build_taiQ(t1, BooQ, contract):
    return contract("ma,miQ->aiQ", t1, BooQ)

#tabQ = sum_m (t1_m^a B_mb^Q)
def build_tabQ(t1, BovQ, contract):
    return contract("ma,mbQ->abQ", t1, BovQ)

# tQ = sum_mf (t1_m^f B_mf^Q)
def build_tQ(t1, BovQ, contract):
    return 2.0 * contract("mf,mfQ->Q", t1, BovQ)

# TiaQ = sum_jb (B_jb^Q u_ij^ab)  //u_ij^ab = 2 * t2_ij^ab - t2_ij^ba
def build_TiaQ(u_ijab, BovQ, contract):
    return contract("ijab,jbQ->iaQ", u_ijab, BovQ)

# TiaQ_tilde = sum_jb (t1_m^a t_im^Q)
def build_TiaQ_tilde(t1, tijQ, contract):
    return contract("ma,imQ->aiQ", t1, tijQ).swapaxes(0,1)

# t'iaQ = tiaQ - taiQ - TiaQ_tilde
def build_tiaQ_prime(tiaQ, taiQ, TiaQ_tilde):
    return tiaQ - taiQ.swapaxes(0,1) - TiaQ_tilde

# Tau_ia^Q = sum_mf ((2 * tau_im^af - tau_mi^af) B_mf^Q)  //tau_ij^ab = t2 + 0.5 * t_i^a t_j^b
def build_Tau_iaQ(tau_ijab, BovQ, contract):
    tau = 2 * tau_ijab - tau_ijab.swapaxes(0,1)
    return contract("imaf,mfQ->iaQ", tau, BovQ)

# Tau'_ia^Q = t_ia^Q + Tau_ia^Q
def build_Tau_iaQ_prime(tiaQ, Tau_iaQ):
    return tiaQ + Tau_iaQ

# Tau'_ia^Q = - t_ai^Q + Tau_ia^Q
def build_Tau_iaQ_double_prime(taiQ, Tau_iaQ):
    return Tau_iaQ - taiQ.swapaxes(0,1)