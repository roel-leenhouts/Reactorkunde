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

# NOTE: No kinetics in this file on purpose.
# Students define rate_constant, reaction_rate, and ODEfun_dXdV in the notebook.

# ------- SOLVER -------
def solve_system_PFR_single(
    P, T, A1, E1, ya0, Fa0, Vspan, initial_conditions,
    rate_constant_fn, reaction_rate_fn, ODEfun_dXdV_fn
):
    """
    Generic PFR solver. Kinetics + ODE callback are injected as callables
    coming from the notebook (student code).
    - ODEfun_dXdV_fn must accept: (X, V, P, T, E1, ya0, Fa0)
      and may internally use student's rate_constant/reaction_rate.
    """
    if initial_conditions is None:
        raise ValueError("initial_conditions must be a tuple/array, e.g. (0.0,) for X_A(0)=0")

    # Integrate student's ODE
    X = odeint(ODEfun_dXdV_fn, np.array(initial_conditions), Vspan, (P, T, E1, ya0, Fa0))

    # Post-processing using the injected kinetics (module remains agnostic)
    Cto = P * 1e5 / (R * T) / 1000.0
    Cao = ya0 * Cto
    epsilon = ya0 * (1 + 1 - 1)  # 0 here for A->B with no net change in moles

    k  = rate_constant_fn(A1, E1, T)
    Ca = Cao * (1 - X) / (1 + epsilon * X)
    ra = -reaction_rate_fn(k, Ca)  # negative for consumption of A (we plot -ra)
    Cb = Cao * X / (1 + epsilon * X)
    Cc = Cb
    return X, Ca, Cb, Cc, ra

# ------- DRAW (re-draws entire figure each time; works with inline) -------
def draw_single(
    Fa0, E1, T, ya0,
    rate_constant_fn,
    reaction_rate_fn,
    ODEfun_dXdV_fn,
    A1_param=A1, P_param=P, Vspan_param=Vspan
):
    """
    draw_single is agnostic to kinetics/math; it just wires callables through.
    Students never see this file; they only provide the functions in the notebook.
    """
    X, Ca, Cb, Cc, ra = solve_system_PFR_single(
        P=P_param, T=T, A1=A1_param, E1=E1, ya0=ya0, Fa0=Fa0, Vspan=Vspan_param,
        initial_conditions=(0.0,),                       # X_A(0) = 0
        rate_constant_fn=rate_constant_fn,
        reaction_rate_fn=reaction_rate_fn,
        ODEfun_dXdV_fn=ODEfun_dXdV_fn
    )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 4.5))
    fs = 10
    V_end = Vspan_param[-1]

    # ax1 off
    ax1.axis("off")

    # Conversion
    ax2.set_prop_cycle(cycler_op)
    ax2.plot(Vspan_param, X)
    ax2.legend([r"$X_{A,end}$" + f"\n= {np.round(float(X[-1][0]), 2)}"],
               loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax2.set_xlabel("V [L]", fontsize=fs); ax2.set_ylabel("Conversion", fontsize=fs)
    ax2.set_ylim(0, 1.0); ax2.set_xlim(0, V_end)

    # Concentrations
    ax3.set_prop_cycle(cycler_op)
    ax3.plot(Vspan_param, Ca, Vspan_param, Cb, Vspan_param, Cc)
    ax3.legend([r"$C_A$", r"$C_B$", r"$C_C$"], loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax3.set_xlabel("V [L]", fontsize=fs); ax3.set_ylabel("Concentration [$mol/L$]", fontsize=fs)
    ax3.set_ylim(0, 0.1); ax3.set_xlim(0, V_end)

    # Rate (plot positive magnitude -ra)
    ax4.set_prop_cycle(cycler_op)
    ax4.plot(Vspan_param, -ra)
    ax4.legend([r"$-r_A$"], loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax4.set_xlabel("V [L]", fontsize=fs); ax4.set_ylabel(r"Reaction Rate [$mol/L\cdot min$]", fontsize=fs)
    ax4.set_ylim(0, 1); ax4.set_xlim(0, V_end)

    plt.tight_layout(h_pad=H_PAD)
    plt.show()

# ---- (Optional) Stubs below are placeholders if you keep series/recycle.
# They deliberately raise to avoid silent NameErrors until implemented.

def solve_series(
    P, T, E1, Eratio, ya0, Fa0, Vspan,
    rate_constant_fn, reaction_rate_fn, ODEfun_dFdV_fn,
    A1_param=A1, A2_param=A2
):
    """
    Series PFR A->B->C with molar flows Y = [Fa, Fb, Fc].
    Kinetics and ODE callback are injected from the notebook.

    ODEfun_dFdV_fn must accept: (Y, V, P, T, E1, Eratio, ya0, Fa0)
    and can use the student rate laws internally.
    """
    y0 = np.array([Fa0, 0.0, 0.0])  # inlet fresh only
    Y = odeint(ODEfun_dFdV_fn, y0, Vspan, (P, T, E1, Eratio, ya0, Fa0))
    Fa, Fb, Fc = Y.T

    # concentrations along reactor (with inert)
    Cto = P * 1e5 / (R * T) / 1000.0
    Ft  = Fa + Fb + Fc + (1 - ya0) / ya0 * Fa0
    Ca, Cb, Cc = Cto * Fa / Ft, Cto * Fb / Ft, Cto * Fc / Ft

    # conversion and yield
    Ca0 = Cto * ya0
    X_A = (Ca0 - Ca) / Ca0
    Y_B = Cb / (Ca0 - Ca)

    # local rates using injected kinetics (same as in the ODE)
    k1 = rate_constant_fn(A1_param, E1, T)
    k2 = rate_constant_fn(A2_param, E1 / Eratio, T)  # keep your E2 = E1/Eratio convention
    r1a = reaction_rate_fn(k1, Ca)          # A -> B (positive magnitude)
    r2b = reaction_rate_fn(k2, Cb)          # B -> C
    ra  = r1a                               # consumption of A (positive magnitude)
    rb  = r1a - r2b                         # net for B
    rc  = r2b                               # formation of C

    return (Y, Ca, Cb, Cc, X_A, Y_B, ra, rb, rc)


def draw_series(
    ya0, T, Eratio, Fa0, E1,
    rate_constant_fn,
    reaction_rate_fn,
    ODEfun_dFdV_fn,
    A1_param=A1, A2_param=A2, P_param=P, Vspan_param=Vspan
):
    (Y, Ca, Cb, Cc, X_A, Y_B, ra, rb, rc) = solve_series(
        P=P_param, T=T, E1=E1, Eratio=Eratio, ya0=ya0, Fa0=Fa0, Vspan=Vspan_param,
        rate_constant_fn=rate_constant_fn,
        reaction_rate_fn=reaction_rate_fn,
        ODEfun_dFdV_fn=ODEfun_dFdV_fn,
        A1_param=A1_param, A2_param=A2_param
    )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 4.5))
    fs = 10; V_end = Vspan_param[-1]
    ax1.axis("off")

    # Conversion & Yield
    ax2.set_prop_cycle(cycler_op)
    ax2.plot(Vspan_param, X_A, Vspan_param, Y_B)
    ax2.legend([r"$X_{A,end}$" + f"\n= {np.round(float(X_A[-1]), 2)}", r"$Y_B$"],
               loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax2.set_xlabel("V [L]", fontsize=fs); ax2.set_ylabel("Conversion and yield [-]", fontsize=fs)
    ax2.set_ylim(0, 1); ax2.set_xlim(0, V_end)

    # Concentrations
    ax3.set_prop_cycle(cycler_op)
    ax3.plot(Vspan_param, Ca, Vspan_param, Cb, Vspan_param, Cc)
    ax3.legend([r"$C_A$", r"$C_B$", r"$C_C$"], loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax3.set_xlabel("V [L]", fontsize=fs); ax3.set_ylabel("Concentration [mol/L]", fontsize=fs)
    ax3.set_xlim(0, V_end)

    # Rates (plot -ra as positive magnitude for A)
    ax4.set_prop_cycle(cycler_op)
    ax4.plot(Vspan_param, -ra, Vspan_param, rb, Vspan_param, rc)
    ax4.legend([r"$-r_A$", r"$r_B$", r"$r_C$"], loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax4.set_xlabel("V [L]", fontsize=fs); ax4.set_ylabel(r"Reaction Rate [mol/L·min]", fontsize=fs)
    ax4.hlines(0, 0, V_end, linestyles="--", colors="black")
    ax4.set_xlim(0, V_end)

    plt.tight_layout(h_pad=H_PAD)
    plt.show()


 def draw_recycle(
    ya0, T, Eratio, Fa0, E1, RR,
    solve_recycle_fn,                  # <-- injected from notebook
    A1_param=A1, A2_param=A2, P_param=P, Vspan_param=Vspan
):
    """
    draw_recycle just calls the provided `solve_recycle_fn` and plots the results.
    The solver lives in the notebook (students can see/edit it).
    """
    (Y, Ca, Cb, Cc, X_A, Y_B, ra, rb, rc) = solve_recycle_fn(
        P=P_param, T=T, E1=E1, Eratio=Eratio, ya0=ya0,
        Fa0=Fa0, RR=RR, Vspan=Vspan_param
    )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8,4.5))
    fs=10; V_end=Vspan_param[-1]
    ax1.axis("off")

    # Conversion & Yield
    ax2.set_prop_cycle(cycler_op)
    ax2.plot(Vspan_param, X_A, Vspan_param, Y_B)
    ax2.legend([r"$X_{A,end}$" + f"\n= {np.round(float(X_A[-1]),2)}", r"$Y_B$"],
               loc="center left", bbox_to_anchor=(1.05,0.5))
    ax2.set_xlabel("V [L]", fontsize=fs); ax2.set_ylabel("Conversion and yield [-]", fontsize=fs)
    ax2.set_ylim(0,1); ax2.set_xlim(0,V_end)

    # Concentrations
    ax3.set_prop_cycle(cycler_op)
    ax3.plot(Vspan_param, Ca, Vspan_param, Cb, Vspan_param, Cc)
    ax3.legend([r"$C_A$", r"$C_B$", r"$C_C$"], loc="center left", bbox_to_anchor=(1.05,0.5))
    ax3.set_xlabel("V [L]", fontsize=fs); ax3.set_ylabel("Concentration [mol/L]", fontsize=fs)
    ax3.set_xlim(0,V_end)

    # Rates
    ax4.set_prop_cycle(cycler_op)
    ax4.plot(Vspan_param, -ra, Vspan_param, rb, Vspan_param, rc)
    ax4.legend([r"$-r_A$", r"$r_B$", r"$r_C$"], loc="center left", bbox_to_anchor=(1.05,0.5))
    ax4.set_xlabel("V [L]", fontsize=fs); ax4.set_ylabel(r"Reaction Rate [mol/L·min]", fontsize=fs)
    ax4.hlines(0,0,V_end, linestyles="--", colors="black")
    ax4.set_xlim(0,V_end)

    plt.tight_layout(h_pad=H_PAD)
    plt.show()

