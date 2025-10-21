# ------- REQUIREMENTS & BACKEND (works with %matplotlib inline) -------
from google.colab import output
output.enable_custom_widget_manager()

# ------- IMPORTS -------
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import ipywidgets as ipw
from cycler import cycler

# ------- STYLE -------
from matplotlib import rc
rc("font", family="serif")
rc("axes", labelsize=12)
kul_1 = "#1FABD5"; kul_2 = "#1D8DB0"; kul_3 = "#116E8A"; kul_4 = "#52BDEC"; kul_5 = "#00407A"
rosso = "#d40000"; arg = "#fc9105"; kulorange = "#DD8A2E"
H_PAD = 1.2

cycler_op = (
    cycler("linestyle", ["--", "dotted", "--", "dotted"])
    + cycler("color", [kul_1, arg, rosso, kul_4])
    + cycler("markersize", [8] * 4)
)

np.seterr(divide='ignore', invalid='ignore')

# ---- Constants / Defaults ----
R = 8.314  # J/mol/K
P = 6      # bar
T0 = 1100  # K
ya0_0 = 1.0 # feed fraction A (default)
Fa0_0 = 193.0    # mol/min
A1 = 1.1034e6    # L/(mol·min)
A2 = A1
E1_0 = 80000.0   # J/mol
Eratio0 = np.exp(-0.4)
V = 2130.0       # L
Vspan = np.linspace(0, V, 200)
RR0   = 0.5

# ------- SOLVER -------
def solve_system_PFR_single(P, T, A1, E1, ya0, Fa0, Vspan, initial_conditions):
    X = odeint(ODEfun_dXdV, np.array(initial_conditions), Vspan, (P, T, E1, ya0, Fa0))
    Cto = P * 1e5 / (R * T) / 1000
    Cao = ya0 * Cto
    epsilon = ya0 * (1 + 1 - 1)
    k = rate_constant(A1, E1, T)
    Ca = Cao * (1 - X) / (1 + epsilon * X)
    ra = -reaction_rate(k, Ca)
    Cb = Cao * X / (1 + epsilon * X)
    Cc = Cb
    return X, Ca, Cb, Cc, ra

# ------- DRAW (re-draws entire figure each time; works with inline) -------
def draw_single(Fa0, E1, T, ya0):
    X, Ca, Cb, Cc, ra = solve_system_PFR_single(
        P=P, T=T, A1=A1, E1=E1, ya0=ya0, Fa0=Fa0, Vspan=Vspan, initial_conditions=(0.0,)
    )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 4.5))
    fs = 10
    V_end = Vspan[-1]

    # ax1 off
    ax1.axis("off")

    # Conversion
    ax2.set_prop_cycle(cycler_op)
    p1 = ax2.plot(Vspan, X)[0]
    ax2.legend([r"$X_{A,end}$" + f"\n= {np.round(float(X[-1][0]), 2)}"], loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax2.set_xlabel("V [L]", fontsize=fs); ax2.set_ylabel("Conversion", fontsize=fs)
    ax2.set_ylim(0, 1.0); ax2.set_xlim(0, V_end)

    # Concentrations
    ax3.set_prop_cycle(cycler_op)
    p2, p3, p4 = ax3.plot(Vspan, Ca, Vspan, Cb, Vspan, Cc)
    ax3.legend([r"$C_A$", r"$C_B$", r"$C_C$"], loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax3.set_xlabel("V [L]", fontsize=fs); ax3.set_ylabel("Concentration [$mol/L$]", fontsize=fs)
    ax3.set_ylim(0, 0.1); ax3.set_xlim(0, V_end)

    # Rate
    ax4.set_prop_cycle(cycler_op)
    p5 = ax4.plot(Vspan, -ra)[0]
    ax4.legend([r"$-r_A$"], loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax4.set_xlabel("V [L]", fontsize=fs); ax4.set_ylabel(r"Reaction Rate [$mol/L\cdot s$]", fontsize=fs)
    ax4.set_ylim(0, 1); ax4.set_xlim(0, V_end)

    plt.tight_layout(h_pad=H_PAD)
    plt.show()

# ---- Solve + post-process ----
def solve_series(P, T, E1, Eratio, ya0, Fa0, Vspan):
    # inlet molar flows (fresh only)
    y0 = np.array([Fa0, 0.0, 0.0])
    Y = odeint(ODEfun_dFdV, y0, Vspan, (P, T, E1, Eratio, ya0, Fa0))
    Fa, Fb, Fc = Y.T
    Cto = P*1e5/(R*T)/1000.0
    Ft  = Fa + Fb + Fc + (1-ya0)/ya0 * Fa0
    Ca, Cb, Cc = Cto*Fa/Ft, Cto*Fb/Ft, Cto*Fc/Ft
    Ca0 = Cto * ya0
    X_A = (Ca0 - Ca)/Ca0
    Y_B = Cb / (Ca0 - Ca)
    k1 = rate_constant(A1, E1, T)
    k2 = rate_constant(A2, E1/Eratio, T)
    r1a = r2(k1, Ca)          # A->B (positive magnitude)
    r2b = r2(k2, Cb)          # B->C
    ra = r1a                 # consumption rate of A (positive magnitude)
    rb = r1a - r2b           # net for B
    rc = r2b                 # formation of C
    return (Y, Ca, Cb, Cc, X_A, Y_B, ra, rb, rc)

# ---- Draw everything (re-renders on slider change) ----
def draw_series(ya0, T, Eratio, Fa0, E1):
    (Y, Ca, Cb, Cc, X_A, Y_B, ra, rb, rc) = solve_series(
        P=P, T=T, E1=E1, Eratio=Eratio, ya0=ya0, Fa0=Fa0, Vspan=Vspan
    )
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8,4.5))
    fs=10; V_end=Vspan[-1]
    ax1.axis("off")

    # Conversion & Yield
    ax2.set_prop_cycle(cycler_op)
    ax2.plot(Vspan, X_A, Vspan, Y_B)
    ax2.legend([r"$X_{A,end}$" + f"\n= {np.round(float(X_A[-1]),2)}", r"$Y_B$"],
               loc="center left", bbox_to_anchor=(1.05,0.5))
    ax2.set_xlabel("V [L]", fontsize=fs); ax2.set_ylabel("Conversion and yield [-]", fontsize=fs)
    ax2.set_ylim(0,1); ax2.set_xlim(0,V_end)

    # Concentrations
    ax3.set_prop_cycle(cycler_op)
    ax3.plot(Vspan, Ca, Vspan, Cb, Vspan, Cc)
    ax3.legend([r"$C_A$", r"$C_B$", r"$C_C$"], loc="center left", bbox_to_anchor=(1.05,0.5))
    ax3.set_xlabel("V [L]", fontsize=fs); ax3.set_ylabel("Concentration [mol/L]", fontsize=fs)
    ax3.set_xlim(0,V_end)

    # Rates
    ax4.set_prop_cycle(cycler_op)
    ax4.plot(Vspan, -ra, Vspan, rb, Vspan, rc)
    ax4.legend([r"$-r_A$", r"$r_B$", r"$r_C$"], loc="center left", bbox_to_anchor=(1.05,0.5))
    ax4.set_xlabel("V [L]", fontsize=fs); ax4.set_ylabel(r"Reaction Rate [mol/L·min]", fontsize=fs)
    ax4.hlines(0,0,V_end, linestyles="--", colors="black")
    ax4.set_xlim(0,V_end)

    plt.tight_layout(h_pad=H_PAD)
    plt.show()

# ---- Draw ----
def draw_recycle(ya0, T, Eratio, Fa0, E1, RR):
    (Y, Ca, Cb, Cc, X_A, Y_B, ra, rb, rc) = solve_recycle(
        P=P, T=T, E1=E1, Eratio=Eratio, ya0=ya0, Fa0=Fa0, RR=RR, Vspan=Vspan
    )
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8,4.5))
    fs=10; V_end=Vspan[-1]
    ax1.axis("off")

    # Conversion & Yield
    ax2.set_prop_cycle(cycler_op)
    ax2.plot(Vspan, X_A, Vspan, Y_B)
    ax2.legend([r"$X_{A,end}$" + f"\n= {np.round(float(X_A[-1]),2)}", r"$Y_B$"],
               loc="center left", bbox_to_anchor=(1.05,0.5))
    ax2.set_xlabel("V [L]", fontsize=fs); ax2.set_ylabel("Conversion and yield [-]", fontsize=fs)
    ax2.set_ylim(0,1); ax2.set_xlim(0,V_end)

    # Concentrations
    ax3.set_prop_cycle(cycler_op)
    ax3.plot(Vspan, Ca, Vspan, Cb, Vspan, Cc)
    ax3.legend([r"$C_A$", r"$C_B$", r"$C_C$"], loc="center left", bbox_to_anchor=(1.05,0.5))
    ax3.set_xlabel("V [L]", fontsize=fs); ax3.set_ylabel("Concentration [mol/L]", fontsize=fs)
    ax3.set_xlim(0,V_end)

    # Rates
    ax4.set_prop_cycle(cycler_op)
    ax4.plot(Vspan, -ra, Vspan, rb, Vspan, rc)
    ax4.legend([r"$-r_A$", r"$r_B$", r"$r_C$"], loc="center left", bbox_to_anchor=(1.05,0.5))
    ax4.set_xlabel("V [L]", fontsize=fs); ax4.set_ylabel(r"Reaction Rate [mol/L·min]", fontsize=fs)
    ax4.hlines(0,0,V_end, linestyles="--", colors="black")
    ax4.set_xlim(0,V_end)

    plt.tight_layout(h_pad=H_PAD)
    plt.show()

