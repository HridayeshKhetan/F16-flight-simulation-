#!/usr/bin/env python3
"""
================================================================================
 F-16 FLIGHT DYNAMICS SIMULATOR
================================================================================
A real-time, 17-state nonlinear F-16 flight dynamics simulation with a
Taichi-accelerated aerodynamic model, a numerically-designed LQR flight
controller, an autopilot maneuver state machine, and a stylized dark/neon
Pygame visualizer (3D flight path, phase-space plots, HUD + control panel).

Run:
    python3 f16_simulator.py

Controls:
    Left-drag   : rotate the 3D camera
    Scroll      : zoom the 3D camera
    Click a button in the control panel to fly a maneuver
================================================================================
"""
import sys
import math

import numpy as np
import taichi as ti
import pygame
from scipy.optimize import fsolve
from scipy.linalg import solve_continuous_are

ti.init(arch=ti.cpu, default_fp=ti.f32)

# ==============================================================================
# 1. PHYSICAL CONSTANTS
# ==============================================================================
MASS = 636.94          # slugs
G0 = 32.17             # ft/s^2
S_WING = 300.0         # ft^2
B_SPAN = 30.0          # ft
C_BAR = 11.32          # ft
HE = 160.0             # lb-ft-s engine angular momentum

C1 = -0.770
C2 = 0.02755
C3 = 1.055e-4
C4 = 1.642e-6
C5 = 0.9604
C6 = 1.759e-2
C7 = 1.792e-5
C8 = -0.7336
C9 = 1.587e-5

THRUST_MAX = 27000.0   # lbf sea-level static (afterburner)

# ==============================================================================
# 2. AERODYNAMIC TABLES 
# ==============================================================================
ALPHA_BP = np.array([-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45], dtype=np.float64)
BETA_BP = np.array([-30, -20, -10, 0, 10, 20, 30], dtype=np.float64)
EL_BP = np.array([-24, -12, 0, 12, 24], dtype=np.float64)
AIL_BP = np.array([-21.5, 0, 21.5], dtype=np.float64)
RUD_BP = np.array([-30, 0, 30], dtype=np.float64)

N_A, N_B, N_E, N_AIL, N_RUD = len(ALPHA_BP), len(BETA_BP), len(EL_BP), len(AIL_BP), len(RUD_BP)


def _cl_lift(alpha_deg):
    a = np.radians(alpha_deg)
    stall = np.exp(-np.clip((alpha_deg - 22.0) / 25.0, 0, None) ** 2)
    return 6.0 * np.sin(a) * stall


def build_tables():
    CL = np.zeros((N_A, N_E))
    for i, a in enumerate(ALPHA_BP):
        base = _cl_lift(a)
        for j, e in enumerate(EL_BP):
            CL[i, j] = base + 0.006 * e

    CD = np.zeros(N_A)
    for i, a in enumerate(ALPHA_BP):
        cl = _cl_lift(a)
        CD[i] = 0.020 + 0.25 * cl ** 2 + 0.00025 * a ** 2

    CM = np.zeros((N_A, N_E))
    for i, a in enumerate(ALPHA_BP):
        base = -0.010 * a
        for j, e in enumerate(EL_BP):
            CM[i, j] = base - 0.016 * e

    CLROLL = np.zeros((N_A, N_B, N_AIL))
    for i, a in enumerate(ALPHA_BP):
        for j, b in enumerate(BETA_BP):
            dihedral = -0.0011 * (1.0 + 0.015 * a) * b
            for k, da in enumerate(AIL_BP):
                ail_power = (0.0013 - 0.0000055 * a) * da
                CLROLL[i, j, k] = dihedral + ail_power

    CN = np.zeros((N_A, N_B, N_RUD))
    for i, a in enumerate(ALPHA_BP):
        for j, b in enumerate(BETA_BP):
            weathercock = 0.0016 * (1.0 + 0.01 * a) * b
            for k, dr in enumerate(RUD_BP):
                rud_power = -0.00105 * dr
                CN[i, j, k] = weathercock + rud_power

    return CL, CD, CM, CLROLL, CN


CL_TAB, CD_TAB, CM_TAB, CLROLL_TAB, CN_TAB = build_tables()

# ==============================================================================
# 3. TAICHI FIELDS
# ==============================================================================
STATE_N = 17
CTRL_N = 4  # elevator(deg), aileron(deg), rudder(deg), throttle_cmd(0-100 pct)

state_field = ti.field(dtype=ti.f32, shape=STATE_N)
deriv_field = ti.field(dtype=ti.f32, shape=STATE_N)
ctrl_field = ti.field(dtype=ti.f32, shape=CTRL_N)

# aux_field: [alpha_deg, beta_deg, Nz(g), Ny(g), Vt(ft/s), mach, qbar, thrust_lbf]
AUX_N = 8
aux_field = ti.field(dtype=ti.f32, shape=AUX_N)
coeff_field = ti.field(dtype=ti.f32, shape=6)  # Cx,Cy,Cz,Cl,Cm,Cn

cl_ti = ti.field(dtype=ti.f32, shape=(N_A, N_E))
cd_ti = ti.field(dtype=ti.f32, shape=(N_A,))
cm_ti = ti.field(dtype=ti.f32, shape=(N_A, N_E))
clroll_ti = ti.field(dtype=ti.f32, shape=(N_A, N_B, N_AIL))
cn_ti = ti.field(dtype=ti.f32, shape=(N_A, N_B, N_RUD))

cl_ti.from_numpy(CL_TAB.astype(np.float32))
cd_ti.from_numpy(CD_TAB.astype(np.float32))
cm_ti.from_numpy(CM_TAB.astype(np.float32))
clroll_ti.from_numpy(CLROLL_TAB.astype(np.float32))
cn_ti.from_numpy(CN_TAB.astype(np.float32))

ALPHA_MIN, ALPHA_DX = float(ALPHA_BP[0]), float(ALPHA_BP[1] - ALPHA_BP[0])
BETA_MIN, BETA_DX = float(BETA_BP[0]), float(BETA_BP[1] - BETA_BP[0])
EL_MIN, EL_DX = float(EL_BP[0]), float(EL_BP[1] - EL_BP[0])
AIL_MIN, AIL_DX = float(AIL_BP[0]), float(AIL_BP[1] - AIL_BP[0])
RUD_MIN, RUD_DX = float(RUD_BP[0]), float(RUD_BP[1] - RUD_BP[0])


# ==============================================================================
# 4. INTERPOLATION (bilinear / trilinear)
# ==============================================================================
@ti.func
def _idx_frac(x, xmin, dx, n):
    fx = (x - xmin) / dx
    i = int(ti.floor(fx))
    if i < 0:
        i = 0
    if i > n - 2:
        i = n - 2
    frac = fx - i
    if frac < 0.0:
        frac = 0.0
    if frac > 1.0:
        frac = 1.0
    return i, frac


@ti.func
def interp1(table: ti.template(), x, xmin, dx, n):
    i, frac = _idx_frac(x, xmin, dx, n)
    return table[i] * (1.0 - frac) + table[i + 1] * frac


@ti.func
def interp2(table: ti.template(), x, y, xmin, dx, nx, ymin, dy, ny):
    ix, fx = _idx_frac(x, xmin, dx, nx)
    iy, fy = _idx_frac(y, ymin, dy, ny)
    v00 = table[ix, iy]
    v10 = table[ix + 1, iy]
    v01 = table[ix, iy + 1]
    v11 = table[ix + 1, iy + 1]
    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    return v0 * (1.0 - fy) + v1 * fy


@ti.func
def interp3(table: ti.template(), x, y, z, xmin, dx, nx, ymin, dy, ny, zmin, dz, nz):
    ix, fx = _idx_frac(x, xmin, dx, nx)
    iy, fy = _idx_frac(y, ymin, dy, ny)
    iz, fz = _idx_frac(z, zmin, dz, nz)
    c000 = table[ix, iy, iz]
    c100 = table[ix + 1, iy, iz]
    c010 = table[ix, iy + 1, iz]
    c110 = table[ix + 1, iy + 1, iz]
    c001 = table[ix, iy, iz + 1]
    c101 = table[ix + 1, iy, iz + 1]
    c011 = table[ix, iy + 1, iz + 1]
    c111 = table[ix + 1, iy + 1, iz + 1]
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


# ==============================================================================
# 5. ATMOSPHERE — 1976 U.S. Standard Atmosphere
# ==============================================================================
# Standard layer definitions (ICAO/NOAA 1976 model), given as geopotential
# base altitude (m), base temperature (K), and lapse rate (K/km) for each of
# the 7 layers from sea level to the mesopause (~86 km). Base pressure at
# each layer boundary is derived* here  by cascading the
# barometric formula layer-by-layer from sea level, so the whole table is
# guaranteed internally consistent with g0/R used elsewhere in the sim.
G0_ATM = 32.17405          # ft/s^2, standard gravity
R_AIR = 1716.3             # ft*lbf/(slug*R), specific gas constant for air
M_TO_FT = 3.280839895
K_TO_R = 1.8
PA_TO_PSF = 0.0208854342   # 1 Pa = 0.0208854 lbf/ft^2

_LAYER_H0_M = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852], dtype=np.float64)
_LAYER_LAPSE_K_KM = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0], dtype=np.float64)
_T0_K, _P0_PA = 288.15, 101325.0

_LAYER_H0_FT = _LAYER_H0_M * M_TO_FT
_LAYER_LAPSE_R_FT = _LAYER_LAPSE_K_KM * K_TO_R / (1000.0 * M_TO_FT)
_LAYER_T0_R = np.zeros(8)
_LAYER_P0_PSF = np.zeros(8)
_LAYER_T0_R[0] = _T0_K * K_TO_R
_LAYER_P0_PSF[0] = _P0_PA * PA_TO_PSF
for _i in range(7):
    _h0, _h1 = _LAYER_H0_FT[_i], _LAYER_H0_FT[_i + 1]
    _L = _LAYER_LAPSE_R_FT[_i]
    _T0i, _P0i = _LAYER_T0_R[_i], _LAYER_P0_PSF[_i]
    _dh = _h1 - _h0
    _T1 = _T0i + _L * _dh
    if abs(_L) > 1e-12:
        _P1 = _P0i * (_T1 / _T0i) ** (-G0_ATM / (R_AIR * _L))
    else:
        _P1 = _P0i * math.exp(-G0_ATM * _dh / (R_AIR * _T0i))
    _LAYER_T0_R[_i + 1] = _T1
    _LAYER_P0_PSF[_i + 1] = _P1

atmo_h0 = ti.field(dtype=ti.f32, shape=8)
atmo_T0 = ti.field(dtype=ti.f32, shape=8)
atmo_P0 = ti.field(dtype=ti.f32, shape=8)
atmo_L = ti.field(dtype=ti.f32, shape=8)
atmo_h0.from_numpy(_LAYER_H0_FT.astype(np.float32))
atmo_T0.from_numpy(_LAYER_T0_R.astype(np.float32))
atmo_P0.from_numpy(_LAYER_P0_PSF.astype(np.float32))
atmo_L.from_numpy(_LAYER_LAPSE_R_FT.astype(np.float32))
ATMO_TOP_FT = float(_LAYER_H0_FT[7])


@ti.func
def atmosphere(alt, Vt):
    a_ = ti.max(0.0, alt)
    if a_ > ATMO_TOP_FT:
        a_ = ATMO_TOP_FT
    idx = 0
    for i in ti.static(range(1, 8)):
        if a_ >= atmo_h0[i]:
            idx = i
    h0 = atmo_h0[idx]
    T0i = atmo_T0[idx]
    P0i = atmo_P0[idx]
    L = atmo_L[idx]
    dh = a_ - h0
    T = T0i + L * dh
    P = 0.0
    if ti.abs(L) > 1e-9:
        P = P0i * (T / T0i) ** (-G0_ATM / (R_AIR * L))
    else:
        P = P0i * ti.exp(-G0_ATM * dh / (R_AIR * T0i))
    rho = P / (R_AIR * T)
    a_snd = ti.sqrt(1.4 * R_AIR * T)
    mach = Vt / a_snd
    qbar = 0.5 * rho * Vt * Vt
    return rho, a_snd, mach, qbar


# ==============================================================================
# 6. AERODYNAMICS + THRUST + FULL DERIVATIVE KERNEL
# ==============================================================================
@ti.func
def compute_aero():
    u, v, w = state_field[0], state_field[1], state_field[2]
    p, q, r = state_field[3], state_field[4], state_field[5]
    Vt = ti.sqrt(u * u + v * v + w * w) + 1e-6
    alpha = ti.atan2(w, u)
    beta = ti.asin(ti.max(-1.0, ti.min(1.0, v / Vt)))
    alpha_deg = alpha * 57.29578
    beta_deg = beta * 57.29578

    el, ail, rud = ctrl_field[0], ctrl_field[1], ctrl_field[2]

    CL = interp2(cl_ti, alpha_deg, el, ALPHA_MIN, ALPHA_DX, N_A, EL_MIN, EL_DX, N_E)
    CD = interp1(cd_ti, alpha_deg, ALPHA_MIN, ALPHA_DX, N_A)
    CM = interp2(cm_ti, alpha_deg, el, ALPHA_MIN, ALPHA_DX, N_A, EL_MIN, EL_DX, N_E)
    ClB = interp3(clroll_ti, alpha_deg, beta_deg, ail, ALPHA_MIN, ALPHA_DX, N_A,
                  BETA_MIN, BETA_DX, N_B, AIL_MIN, AIL_DX, N_AIL)
    CN = interp3(cn_ti, alpha_deg, beta_deg, rud, ALPHA_MIN, ALPHA_DX, N_A,
                 BETA_MIN, BETA_DX, N_B, RUD_MIN, RUD_DX, N_RUD)

    CD = CD + 0.00025 * el * el + 0.00012 * ail * ail + 0.00012 * rud * rud

    sa, ca = ti.sin(alpha), ti.cos(alpha)
    Cx = CL * sa - CD * ca
    Cz = -CL * ca - CD * sa

    CY = -0.98 * ti.sin(beta) + 0.0035 * rud

    rho, a_snd, mach, qbar = atmosphere(-state_field[12], Vt)

    Cm = CM - 12.0 * (q * C_BAR / (2.0 * Vt))
    Cl_ = ClB - 0.40 * (p * B_SPAN / (2.0 * Vt))
    Cn_ = CN - 0.35 * (r * B_SPAN / (2.0 * Vt))
    Cy_ = CY - 0.03 * (p * B_SPAN / (2.0 * Vt)) + 0.20 * (r * B_SPAN / (2.0 * Vt))

    coeff_field[0] = Cx
    coeff_field[1] = Cy_
    coeff_field[2] = Cz
    coeff_field[3] = Cl_
    coeff_field[4] = Cm
    coeff_field[5] = Cn_

    Nz = -Cz * qbar * S_WING / (MASS * G0)
    Ny = Cy_ * qbar * S_WING / (MASS * G0)

    aux_field[0] = alpha_deg
    aux_field[1] = beta_deg
    aux_field[2] = Nz
    aux_field[3] = Ny
    aux_field[4] = Vt
    aux_field[5] = mach
    aux_field[6] = qbar
    return qbar, mach


@ti.func
def thrust_lbf(P_pct, mach, alt):
    lapse = ti.exp(-alt / 45000.0) * (1.0 - 0.15 * mach)
    if lapse < 0.15:
        lapse = 0.15
    return THRUST_MAX * (P_pct / 100.0) * lapse


@ti.kernel
def compute_derivatives():
    u, v, w = state_field[0], state_field[1], state_field[2]
    p, q, r = state_field[3], state_field[4], state_field[5]
    q0, q1, q2, q3 = state_field[6], state_field[7], state_field[8], state_field[9]
    ze = state_field[12]
    P = state_field[13]
    alt = -ze
    if alt < 0.0:
        alt = 0.0

    qbar, mach = compute_aero()
    Cx, Cy, Cz = coeff_field[0], coeff_field[1], coeff_field[2]
    Cl_, Cm, Cn_ = coeff_field[3], coeff_field[4], coeff_field[5]

    T = thrust_lbf(P, mach, alt)
    aux_field[7] = T

    # Euler angles from quaternion (321 sequence), used only for gravity components
    phi = ti.atan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
    s_theta = 2.0 * (q0 * q2 - q3 * q1)
    if s_theta > 1.0:
        s_theta = 1.0
    if s_theta < -1.0:
        s_theta = -1.0
    theta = ti.asin(s_theta)

    # 6-DOF translational equations (body axis)
    udot = r * v - q * w - G0 * ti.sin(theta) + qbar * S_WING * Cx / MASS + T / MASS
    vdot = p * w - r * u + G0 * ti.cos(theta) * ti.sin(phi) + qbar * S_WING * Cy / MASS
    wdot = q * u - p * v + G0 * ti.cos(theta) * ti.cos(phi) + qbar * S_WING * Cz / MASS

    # rotational rate equations
    pdot = (C2 * p + C1 * r + C4 * HE) * q + qbar * S_WING * B_SPAN * (C3 * Cl_ + C4 * Cn_)
    qdot = (C5 * p - C7 * HE) * r + C6 * (r * r - p * p) + qbar * S_WING * C_BAR * C7 * Cm
    rdot = (C8 * p - C2 * r + C9 * HE) * q + qbar * S_WING * B_SPAN * (C4 * Cl_ + C9 * Cn_)

    # quaternion kinematics
    q0dot = 0.5 * (-q1 * p - q2 * q - q3 * r)
    q1dot = 0.5 * (q0 * p + q2 * r - q3 * q)
    q2dot = 0.5 * (q0 * q - q1 * r + q3 * p)
    q3dot = 0.5 * (q0 * r + q1 * q - q2 * p)

    # position kinematics (earth frame)
    xedot = (1 - 2 * (q2 * q2 + q3 * q3)) * u + 2 * (q1 * q2 - q0 * q3) * v + 2 * (q1 * q3 + q0 * q2) * w
    yedot = 2 * (q1 * q2 + q0 * q3) * u + (1 - 2 * (q1 * q1 + q3 * q3)) * v + 2 * (q2 * q3 - q0 * q1) * w
    zedot = 2 * (q1 * q3 - q0 * q2) * u + 2 * (q2 * q3 + q0 * q1) * v + (1 - 2 * (q1 * q1 + q2 * q2)) * w

    Pc = ctrl_field[3]
    Pdot = (Pc - P) / 1.0

    deriv_field[0] = udot
    deriv_field[1] = vdot
    deriv_field[2] = wdot
    deriv_field[3] = pdot
    deriv_field[4] = qdot
    deriv_field[5] = rdot
    deriv_field[6] = q0dot
    deriv_field[7] = q1dot
    deriv_field[8] = q2dot
    deriv_field[9] = q3dot
    deriv_field[10] = xedot
    deriv_field[11] = yedot
    deriv_field[12] = zedot
    deriv_field[13] = Pdot
    deriv_field[14] = 0.0
    deriv_field[15] = 0.0
    deriv_field[16] = 0.0


def set_state(state):
    for i in range(STATE_N):
        state_field[i] = float(state[i])


def set_ctrl(ctrl):
    for i in range(CTRL_N):
        ctrl_field[i] = float(ctrl[i])


def get_deriv():
    return np.array([deriv_field[i] for i in range(STATE_N)])


def get_aux():
    return np.array([aux_field[i] for i in range(AUX_N)])


def f16_derivatives(state, ctrl):
    set_state(state)
    set_ctrl(ctrl)
    compute_derivatives()
    return get_deriv(), get_aux()


# ------------------------------------------------------------------
# Fast path: the whole 4-stage RK4 step fused into ONE Taichi kernel.
# (Calling the Python-level derivative wrapper 4x/step costs ~0.1ms per
#  Python<->Taichi crossing regardless of payload size, so doing this in
#  Python was ~1.8ms/step; fusing it into one kernel call brings that
#  down to ~0.03ms/step -- essential for sustaining 200 Hz physics
#  alongside 60 FPS rendering.)
# ------------------------------------------------------------------
Vec17 = ti.types.vector(17, ti.f32)
Vec4 = ti.types.vector(4, ti.f32)


@ti.func
def eom_vec(st: Vec17, ct: Vec4):
    u, v, w = st[0], st[1], st[2]
    p, q, r = st[3], st[4], st[5]
    q0, q1, q2, q3 = st[6], st[7], st[8], st[9]
    ze = st[12]
    P = st[13]
    alt = ti.max(0.0, -ze)
    Vt = ti.sqrt(u * u + v * v + w * w) + 1e-6
    alpha = ti.atan2(w, u)
    beta = ti.asin(ti.max(-1.0, ti.min(1.0, v / Vt)))
    alpha_deg = alpha * 57.29578
    beta_deg = beta * 57.29578

    el, ail, rud = ct[0], ct[1], ct[2]
    CL = interp2(cl_ti, alpha_deg, el, ALPHA_MIN, ALPHA_DX, N_A, EL_MIN, EL_DX, N_E)
    CD = interp1(cd_ti, alpha_deg, ALPHA_MIN, ALPHA_DX, N_A)
    CM = interp2(cm_ti, alpha_deg, el, ALPHA_MIN, ALPHA_DX, N_A, EL_MIN, EL_DX, N_E)
    ClB = interp3(clroll_ti, alpha_deg, beta_deg, ail, ALPHA_MIN, ALPHA_DX, N_A,
                  BETA_MIN, BETA_DX, N_B, AIL_MIN, AIL_DX, N_AIL)
    CN = interp3(cn_ti, alpha_deg, beta_deg, rud, ALPHA_MIN, ALPHA_DX, N_A,
                 BETA_MIN, BETA_DX, N_B, RUD_MIN, RUD_DX, N_RUD)
    CD = CD + 0.00025 * el * el + 0.00012 * ail * ail + 0.00012 * rud * rud

    sa, ca = ti.sin(alpha), ti.cos(alpha)
    Cx = CL * sa - CD * ca
    Cz = -CL * ca - CD * sa
    CY = -0.98 * ti.sin(beta) + 0.0035 * rud

    rho, a_snd, mach, qbar = atmosphere(alt, Vt)
    Cm = CM - 12.0 * (q * C_BAR / (2.0 * Vt))
    Cl_ = ClB - 0.40 * (p * B_SPAN / (2.0 * Vt))
    Cn_ = CN - 0.35 * (r * B_SPAN / (2.0 * Vt))
    Cy_ = CY - 0.03 * (p * B_SPAN / (2.0 * Vt)) + 0.20 * (r * B_SPAN / (2.0 * Vt))

    T = thrust_lbf(P, mach, alt)

    phi = ti.atan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
    s_theta = ti.max(-1.0, ti.min(1.0, 2.0 * (q0 * q2 - q3 * q1)))
    theta = ti.asin(s_theta)

    udot = r * v - q * w - G0 * ti.sin(theta) + qbar * S_WING * Cx / MASS + T / MASS
    vdot = p * w - r * u + G0 * ti.cos(theta) * ti.sin(phi) + qbar * S_WING * Cy_ / MASS
    wdot = q * u - p * v + G0 * ti.cos(theta) * ti.cos(phi) + qbar * S_WING * Cz / MASS

    pdot = (C2 * p + C1 * r + C4 * HE) * q + qbar * S_WING * B_SPAN * (C3 * Cl_ + C4 * Cn_)
    qdot = (C5 * p - C7 * HE) * r + C6 * (r * r - p * p) + qbar * S_WING * C_BAR * C7 * Cm
    rdot = (C8 * p - C2 * r + C9 * HE) * q + qbar * S_WING * B_SPAN * (C4 * Cl_ + C9 * Cn_)

    q0dot = 0.5 * (-q1 * p - q2 * q - q3 * r)
    q1dot = 0.5 * (q0 * p + q2 * r - q3 * q)
    q2dot = 0.5 * (q0 * q - q1 * r + q3 * p)
    q3dot = 0.5 * (q0 * r + q1 * q - q2 * p)

    xedot = (1 - 2 * (q2 * q2 + q3 * q3)) * u + 2 * (q1 * q2 - q0 * q3) * v + 2 * (q1 * q3 + q0 * q2) * w
    yedot = 2 * (q1 * q2 + q0 * q3) * u + (1 - 2 * (q1 * q1 + q3 * q3)) * v + 2 * (q2 * q3 - q0 * q1) * w
    zedot = 2 * (q1 * q3 - q0 * q2) * u + 2 * (q2 * q3 + q0 * q1) * v + (1 - 2 * (q1 * q1 + q2 * q2)) * w

    Pc = ct[3]
    Pdot = (Pc - P) / 1.0

    return Vec17([udot, vdot, wdot, pdot, qdot, rdot, q0dot, q1dot, q2dot, q3dot,
                  xedot, yedot, zedot, Pdot, 0.0, 0.0, 0.0])


@ti.kernel
def rk4_integrate(dt: ti.f32):
    st0 = Vec17([state_field[i] for i in range(17)])
    ct = Vec4([ctrl_field[i] for i in range(4)])
    k1 = eom_vec(st0, ct)
    k2 = eom_vec(st0 + dt / 2 * k1, ct)
    k3 = eom_vec(st0 + dt / 2 * k2, ct)
    k4 = eom_vec(st0 + dt * k3, ct)
    st1 = st0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    qn = ti.sqrt(st1[6] ** 2 + st1[7] ** 2 + st1[8] ** 2 + st1[9] ** 2)
    if qn > 1e-9:
        for i in ti.static(range(6, 10)):
            st1[i] = st1[i] / qn
    for i in ti.static(range(17)):
        state_field[i] = st1[i]


# ==============================================================================
# 7. TRIM SOLVER
# ==============================================================================
def _state_from_trim(Vt, alpha_deg, alt):
    alpha = np.radians(alpha_deg)
    s = np.zeros(17)
    s[0] = Vt * np.cos(alpha)
    s[2] = Vt * np.sin(alpha)
    s[6] = 1.0
    s[12] = -alt
    return s


def _trim_residual(x, Vt, alt):
    alpha_deg, el, P = x
    s = _state_from_trim(Vt, alpha_deg, alt)
    s[13] = P
    ctrl = [el, 0.0, 0.0, P]
    d, aux = f16_derivatives(s, ctrl)
    return [d[0], d[2], d[4]]  # udot, wdot, qdot -> 0


def find_trim(Vt=502.0, alt=1000.0, guess=(2.2, -2.0, 55.0)):
    """Solve for (alpha, elevator, throttle) that gives steady, wings-level flight."""
    sol, info, ier, msg = fsolve(_trim_residual, guess, args=(Vt, alt), full_output=True,
                                  xtol=1e-10, maxfev=5000, epsfcn=1e-3)
    resid_norm = float(np.linalg.norm(info["fvec"]))
    if resid_norm > 0.05:
        print(f"[trim] warning: residual norm={resid_norm:.4g} ({msg.strip()})")
    alpha_deg, el, P = sol
    state = _state_from_trim(Vt, alpha_deg, alt)
    state[13] = P
    ctrl = np.array([el, 0.0, 0.0, P])
    d, aux = f16_derivatives(state, ctrl)
    return state, ctrl, d, aux


# ==============================================================================
# 8. LQR DESIGN (numerically linearized from the actual Taichi aero model)
# ==============================================================================
def _jac(func, x0, eps=1e-4):
    n = len(x0)
    f0 = np.array(func(x0))
    m = len(f0)
    J = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        step = eps * max(1.0, abs(x0[i]))
        dx[i] = step
        f_p = np.array(func(x0 + dx))
        f_m = np.array(func(x0 - dx))
        J[:, i] = (f_p - f_m) / (2 * step)
    return J


def design_longitudinal_lqr(trim_state, trim_ctrl):
    """State feedback on [u, w, q, integ_Nz] -> elevator perturbation."""
    u0, w0, q0 = trim_state[0], trim_state[2], trim_state[4]
    el0 = trim_ctrl[0]

    def dyn(x):
        u, w, q, el = x
        s = trim_state.copy()
        s[0], s[2], s[4] = u, w, q
        c = trim_ctrl.copy()
        c[0] = el
        d, aux = f16_derivatives(s, c)
        return [d[0], d[2], d[4], aux[2]]

    x0 = np.array([u0, w0, q0, el0])
    J = _jac(dyn, x0)
    A3, B3 = J[0:3, 0:3], J[0:3, 3:4]
    dNz_dx, dNz_du = J[3, 0:3], J[3, 3]

    A = np.zeros((4, 4)); A[0:3, 0:3] = A3; A[3, 0:3] = dNz_dx
    B = np.zeros((4, 1)); B[0:3, 0] = B3[:, 0]; B[3, 0] = dNz_du

    Q = np.diag([0.01, 0.01, 1.0, 8.0])
    R = np.array([[1.0]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K.flatten(), (u0, w0, q0, el0)


def design_lateral_lqr(trim_state, trim_ctrl):
    """State feedback on [v, p, r, integ_ps, integ_Nyr] -> [aileron, rudder] perturbation."""
    alpha0 = np.arctan2(trim_state[2], trim_state[0])

    def dyn(x):
        v, p, r, ail, rud = x
        s = trim_state.copy()
        s[1], s[3], s[5] = v, p, r
        c = trim_ctrl.copy()
        c[1], c[2] = ail, rud
        d, aux = f16_derivatives(s, c)
        ps = p * np.cos(alpha0) + r * np.sin(alpha0)
        nyr = aux[3] + r
        return [d[1], d[3], d[5], ps, nyr]

    x0 = np.array([0.0, 0.0, 0.0, trim_ctrl[1], trim_ctrl[2]])
    J = _jac(dyn, x0)
    A3, B2 = J[0:3, 0:3], J[0:3, 3:5]
    dps_dx, dps_du = J[3, 0:3], J[3, 3:5]
    dnyr_dx, dnyr_du = J[4, 0:3], J[4, 3:5]

    A = np.zeros((5, 5)); A[0:3, 0:3] = A3; A[3, 0:3] = dps_dx; A[4, 0:3] = dnyr_dx
    B = np.zeros((5, 2)); B[0:3, :] = B2; B[3, :] = dps_du; B[4, :] = dnyr_du

    Q = np.diag([0.05, 0.3, 0.3, 6.0, 4.0])
    R = np.diag([1.0, 1.0])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K  # 2x5


# ==============================================================================
# 9. AUTOPILOT MANEUVER STATE MACHINE
# ==============================================================================
MANEUVERS = [
    "Steady Flight",
    "Aileron Roll",
    "Barrel Roll",
    "Immelmann Turn",
    "Split-S",
    "Kvochur's Bell",
    "GCAS Recovery",
]

TWO_PI = 2 * np.pi


class Autopilot:
    """Converts a selected maneuver into low-level LQR reference commands
    (Nz_ref, ps_ref, Ny+r_ref, throttle_pct), using state feedback to decide
    phase transitions rather than open-loop timers alone."""

    def __init__(self, trim_throttle):
        self.trim_throttle = trim_throttle
        self.mode = "Steady Flight"
        self.phase = 0
        self.t_phase = 0.0
        self.roll_accum = 0.0
        self.pitch_accum = 0.0
        self.status_text = "Steady & Level"
        self.dived = False

    def start(self, mode):
        if mode not in MANEUVERS:
            return
        self.mode = mode
        self.phase = 0
        self.t_phase = 0.0
        self.roll_accum = 0.0
        self.pitch_accum = 0.0
        self.dived = False

    def _finish(self):
        self.mode = "Steady Flight"
        self.phase = 0
        self.t_phase = 0.0
        self.roll_accum = 0.0
        self.pitch_accum = 0.0
        self.dived = False

    def update(self, s, aux, dt):
        self.t_phase += dt
        alpha_deg, beta_deg, Nz, Ny, Vt, mach, qbar, T = aux
        p, q, r = s[3], s[4], s[5]
        q0, q1, q2, q3 = s[6], s[7], s[8], s[9]
        phi = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        s_theta = np.clip(2 * (q0 * q2 - q3 * q1), -1, 1)
        theta = np.arcsin(s_theta)

        self.roll_accum += p * dt
        self.pitch_accum += q * dt

        nz_ref, ps_ref, nyr_ref, throttle = 1.0, 0.0, 0.0, self.trim_throttle
        status = "Steady & Level"
        m = self.mode

        if m == "Steady Flight":
            status = "Steady & Level"

        elif m == "Aileron Roll":
            ps_ref, nz_ref = 3.0, 1.0
            status = f"Aileron Roll ({np.degrees(self.roll_accum):5.0f} deg)"
            if abs(self.roll_accum) >= TWO_PI:
                self._finish()

        elif m == "Barrel Roll":
            ps_ref, nz_ref = 2.0, 2.3
            status = f"Barrel Roll ({np.degrees(self.roll_accum):5.0f} deg)"
            if abs(self.roll_accum) >= TWO_PI:
                self._finish()

        elif m == "Immelmann Turn":
            if self.phase == 0:
                nz_ref, ps_ref, throttle = 4.0, 0.0, 100.0
                status = f"Immelmann: pulling up ({np.degrees(self.pitch_accum):5.0f} deg)"
                if self.pitch_accum >= np.pi * 0.98:
                    self.phase = 1
                    self.roll_accum = 0.0
            elif self.phase == 1:
                nz_ref, ps_ref = 1.0, 3.0
                status = f"Immelmann: rolling upright ({np.degrees(self.roll_accum):5.0f} deg)"
                if abs(self.roll_accum) >= np.pi * 0.98:
                    self._finish()

        elif m == "Split-S":
            if self.phase == 0:
                nz_ref, ps_ref = 1.0, 3.0
                status = f"Split-S: inverting ({np.degrees(self.roll_accum):5.0f} deg)"
                if abs(self.roll_accum) >= np.pi * 0.98:
                    self.phase = 1
                    self.pitch_accum = 0.0
            elif self.phase == 1:
                nz_ref, throttle = 4.0, 60.0
                status = f"Split-S: diving through ({np.degrees(self.pitch_accum):5.0f} deg)"
                if self.pitch_accum >= np.pi * 0.98:
                    self._finish()

        elif m == "Kvochur's Bell":
            if self.phase == 0:
                nz_ref, throttle = 5.5, 100.0
                status = f"Bell: vertical pull ({np.degrees(self.pitch_accum):5.0f} deg)"
                if self.pitch_accum >= np.pi / 2 * 0.97:
                    self.phase = 1
            elif self.phase == 1:
                nz_ref, throttle = 0.3, 0.0
                status = f"Bell: decelerating (Vt={Vt:5.0f} ft/s)"
                if Vt < 90.0:
                    self.phase = 2
                    self.pitch_accum = 0.0
            elif self.phase == 2:
                nz_ref, throttle = 0.6, 20.0
                status = "Bell: tail-slide pitch-over"
                if self.pitch_accum >= np.radians(25):
                    self.phase = 3
            elif self.phase == 3:
                nz_ref, throttle = 3.0, 100.0
                status = f"Bell: recovery dive (theta={np.degrees(theta):5.0f} deg)"
                if theta < np.radians(-20):
                    self.dived = True
                if self.dived and theta > np.radians(-10):
                    self._finish()

        elif m == "GCAS Recovery":
            if self.phase == 0:
                ps_ref = float(np.clip(-3.0 * phi, -6.0, 6.0))
                nz_ref, throttle = 1.0, 100.0
                status = f"GCAS: wings level (bank={np.degrees(phi):5.0f} deg)"
                if abs(phi) < np.radians(4):
                    self.phase = 1
            elif self.phase == 1:
                nz_ref, throttle = 5.0, 100.0
                status = f"GCAS: pulling up (theta={np.degrees(theta):5.0f} deg)"
                if theta > np.radians(8):
                    self._finish()

        self.status_text = status
        return nz_ref, ps_ref, nyr_ref, throttle, status


# ==============================================================================
# 10. SIMULATION CLASS (RK4 integration @ 200 Hz + LQR feedback)
# ==============================================================================
class F16Simulation:
    def __init__(self, Vt=800.0, alt=12000.0):
        print("[sim] solving trim...")
        state, ctrl, d, aux = find_trim(Vt, alt)
        print(f"[sim] trim: alpha={np.degrees(np.arctan2(state[2], state[0])):.2f} deg, "
              f"elevator={ctrl[0]:.2f} deg, throttle={ctrl[3]:.1f}%")
        print("[sim] designing LQR gains...")
        self.Klon, (self.u0, self.w0, self.q0trim, self.el0) = design_longitudinal_lqr(state, ctrl)
        self.Klat = design_lateral_lqr(state, ctrl)
        print("[sim] Klon =", np.round(self.Klon, 4))
        print("[sim] Klat =\n", np.round(self.Klat, 4))

        self.trim_state = state.copy()
        self.trim_ctrl = ctrl.copy()
        self.s = state.copy()
        self.integ_nz = 0.0
        self.integ_ps = 0.0
        self.integ_nyr = 0.0
        self.t = 0.0
        self.dt = 1.0 / 200.0
        self.autopilot = Autopilot(ctrl[3])
        self.last_aux = aux
        self.last_ctrl_out = (self.el0, 0.0, 0.0, ctrl[3])

        # trail history for the 3D viewport: (xe, ye, ze, Vt)
        self.trail = []
        self.trail_max_len = 1200
        self.trail_accum = 0.0

        # crash / ground-collision state
        self.crashed = False
        self.crash_flash = 0.0
        self.explosion_particles = []
        self._rng = np.random.default_rng()

        # seed the persistent Taichi fields (state_field/ctrl_field are the
        # source of truth for the fast fused-kernel integrator in step())
        set_state(self.s)
        set_ctrl(np.array(self.last_ctrl_out, dtype=np.float32))

    def reset(self):
        """Respawn back at the trimmed spawn condition after a crash."""
        self.s = self.trim_state.copy()
        self.integ_nz = self.integ_ps = self.integ_nyr = 0.0
        self.t = 0.0
        self.last_ctrl_out = (self.el0, 0.0, 0.0, float(self.trim_ctrl[3]))
        self.trail = []
        self.trail_accum = 0.0
        self.crashed = False
        self.crash_flash = 0.0
        self.explosion_particles = []
        self.autopilot.start("Steady Flight")
        set_state(self.s)
        set_ctrl(np.array(self.last_ctrl_out, dtype=np.float32))
        compute_derivatives()
        self.last_aux = get_aux()

    def _trigger_crash(self, s):
        self.crashed = True
        self.crash_flash = 1.0
        s = s.copy()
        s[12] = 0.0       # pin to ground level (ze=0 -> alt=0)
        s[0:6] = 0.0      # zero velocities/rates so the wreck doesn't keep moving
        self.s = s
        set_state(s)
        self.explosion_particles = []
        for _ in range(70):
            theta = self._rng.uniform(0, 2 * np.pi)
            elev = self._rng.uniform(0.05, np.pi / 2)
            speed = self._rng.uniform(20, 140)
            vx = speed * np.cos(elev) * np.cos(theta)
            vy = speed * np.cos(elev) * np.sin(theta)
            vz = -speed * np.sin(elev)  # negative = upward (ze is down-positive)
            life = self._rng.uniform(0.6, 1.8)
            self.explosion_particles.append([0.0, 0.0, 0.0, vx, vy, vz, life, life])

    def _update_particles(self, dt):
        self.crash_flash = max(0.0, self.crash_flash - dt * 1.2)
        alive = []
        for p in self.explosion_particles:
            p[0] += p[3] * dt
            p[1] += p[4] * dt
            p[2] += p[5] * dt
            p[5] += G0 * dt  # gravity (z down-positive)
            p[6] -= dt
            if p[6] > 0:
                alive.append(p)
        self.explosion_particles = alive

    def step(self):
        if self.crashed:
            self._update_particles(self.dt)
            self.t += self.dt
            return "CRASHED — press R to respawn"

        dt = self.dt

        # 1) sensor/feedback read: aux at the current state, using the
        #    previously-commanded control surfaces (still resident in ctrl_field)
        compute_derivatives()
        aux = get_aux()
        self.s = state_field.to_numpy()

        nz_ref, ps_ref, nyr_ref, throttle, status = self.autopilot.update(self.s, aux, dt)

        alpha_deg, beta_deg, Nz, Ny, Vt, mach, qbar, T = aux
        p, q, r = self.s[3], self.s[4], self.s[5]
        alpha = np.radians(alpha_deg)
        ps = p * np.cos(alpha) + r * np.sin(alpha)
        nyr = Ny + r

        self.integ_nz = np.clip(self.integ_nz + (Nz - nz_ref) * dt, -50, 50)
        self.integ_ps = np.clip(self.integ_ps + (ps - ps_ref) * dt, -50, 50)
        self.integ_nyr = np.clip(self.integ_nyr + (nyr - nyr_ref) * dt, -50, 50)

        xlon = np.array([self.s[0] - self.u0, self.s[2] - self.w0, q - self.q0trim, self.integ_nz])
        d_el = float(-self.Klon @ xlon)
        el_cmd = float(np.clip(self.el0 + d_el, -25, 25))

        xlat = np.array([self.s[1], p, r, self.integ_ps, self.integ_nyr])
        d_ail_rud = -self.Klat @ xlat
        ail_cmd = float(np.clip(d_ail_rud[0], -21.5, 21.5))
        rud_cmd = float(np.clip(d_ail_rud[1], -30, 30))

        # 2) push the freshly-computed control surfaces and integrate one
        #    full RK4 step in a single fused Taichi kernel call
        ctrl_vec = np.array([el_cmd, ail_cmd, rud_cmd, throttle], dtype=np.float32)
        set_ctrl(ctrl_vec)
        rk4_integrate(dt)

        s_new = state_field.to_numpy()
        if -s_new[12] <= 0.0:
            self._trigger_crash(s_new)
            return "CRASHED — press R to respawn"

        self.t += dt
        self.last_aux = aux
        self.last_ctrl_out = (el_cmd, ail_cmd, rud_cmd, throttle)

        self.trail_accum += dt
        if self.trail_accum >= 0.02:  # ~50 trail samples/sec is plenty for a smooth ribbon
            self.trail_accum = 0.0
            self.trail.append((s_new[10], s_new[11], s_new[12], Vt))
            if len(self.trail) > self.trail_max_len:
                self.trail.pop(0)

        return status


# ==============================================================================
# 11. VISUALIZATION — dark/neon Pygame dashboard
# ==============================================================================
WIDTH, HEIGHT = 1500, 900
VIEW3D_RECT = pygame.Rect(0, 0, 900, 900)
PHASE_RECT = pygame.Rect(900, 0, 600, 380)
HUD_RECT = pygame.Rect(900, 380, 600, 340)
PANEL_RECT = pygame.Rect(900, 720, 600, 180)

BG_COLOR = (5, 5, 8)
PANEL_BG = (8, 10, 14)
GRID_COLOR = (35, 45, 55)
TEXT_COLOR = (170, 245, 230)
DIM_TEXT = (90, 130, 125)
VIOLET = (178, 102, 255)
TEAL = (64, 224, 208)
NEON_GREEN = (57, 255, 90)
YELLOW = (255, 230, 60)
AMBER = (255, 176, 40)
RED = (255, 70, 70)
CYAN = (60, 230, 255)
BORDER = (50, 220, 200)


def speed_color(t):
    t = max(0.0, min(1.0, t))
    stops = [VIOLET, TEAL, NEON_GREEN, YELLOW]
    seg = t * (len(stops) - 1)
    i = min(int(seg), len(stops) - 2)
    f = seg - i
    c0, c1 = stops[i], stops[i + 1]
    return tuple(int(c0[k] + (c1[k] - c0[k]) * f) for k in range(3))


def quat_to_R(q0, q1, q2, q3):
    return np.array([
        [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)],
    ])


class GlowLayer:
    """A transparent scratch surface for drawing translucent-halo 'neon glow' lines."""

    def __init__(self, size):
        self.surf = pygame.Surface(size, pygame.SRCALPHA)

    def clear(self):
        self.surf.fill((0, 0, 0, 0))

    def line(self, color, p1, p2, width=2, glow=4, glow_alpha=70):
        r, g, b = color[:3]
        pygame.draw.line(self.surf, (r, g, b, glow_alpha), p1, p2, width + glow * 2)
        pygame.draw.line(self.surf, (r, g, b, 255), p1, p2, max(1, width))

    def circle(self, color, center, radius, glow=5, glow_alpha=70):
        r, g, b = color[:3]
        pygame.draw.circle(self.surf, (r, g, b, glow_alpha), center, radius + glow)
        pygame.draw.circle(self.surf, (r, g, b, 255), center, radius)

    def blit_to(self, dest, pos):
        dest.blit(self.surf, pos)


class Visualizer:
    def __init__(self):
        pygame.font.init()
        font_path = pygame.font.match_font('couriernew,dejavusansmono,freemono,monospace') or None
        self.font_sm = pygame.font.Font(font_path, 13)
        self.font_md = pygame.font.Font(font_path, 16)
        self.font_lg = pygame.font.Font(font_path, 24)
        self.font_xl = pygame.font.Font(font_path, 34)

        self.cam_yaw = 0.7
        self.cam_pitch = 0.30
        self.cam_dist = 700.0
        self.dragging = False
        self.last_mouse = (0, 0)

        self.glow3d = GlowLayer(VIEW3D_RECT.size)
        self.glow_phase = GlowLayer(PHASE_RECT.size)

        self.phase_hist = []
        self.phase_hist_max = 260

        self.buttons = []  # filled by draw_panel: list of (rect, label)

    # ---------------- input ----------------
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and VIEW3D_RECT.collidepoint(event.pos):
                self.dragging = True
                self.last_mouse = event.pos
            if event.button == 4:  # scroll up = zoom in
                self.cam_dist = max(80.0, self.cam_dist * 0.9)
            if event.button == 5:  # scroll down = zoom out
                self.cam_dist = min(8000.0, self.cam_dist * 1.1)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                dx = event.pos[0] - self.last_mouse[0]
                dy = event.pos[1] - self.last_mouse[1]
                self.cam_yaw += dx * 0.008
                self.cam_pitch = float(np.clip(self.cam_pitch - dy * 0.008, -1.45, 1.45))
                self.last_mouse = event.pos
        elif event.type == pygame.MOUSEWHEEL:
            self.cam_dist = float(np.clip(self.cam_dist * (0.9 if event.y > 0 else 1.1), 80.0, 8000.0))

    def button_clicked(self, pos):
        for rect, label in self.buttons:
            if rect.collidepoint(pos):
                return label
        return None

    # ---------------- 3D projection ----------------
    def _project(self, dx, dy, dz, focal=520.0):
        cy, sy = math.cos(self.cam_yaw), math.sin(self.cam_yaw)
        x1 = dx * cy - dz * sy
        z1 = dx * sy + dz * cy
        y1 = dy
        cp, sp = math.cos(self.cam_pitch), math.sin(self.cam_pitch)
        y2 = y1 * cp - z1 * sp
        z2 = y1 * sp + z1 * cp
        x2 = x1
        zf = z2 + self.cam_dist
        if zf < 5.0:
            return None
        scale = focal / zf
        cx = VIEW3D_RECT.width / 2
        cy_ = VIEW3D_RECT.height / 2
        return float(cx + x2 * scale), float(cy_ - y2 * scale), float(zf)

    def draw_text(self, surf, text, pos, font=None, color=TEXT_COLOR):
        font = font or self.font_sm
        surf.blit(font.render(text, True, color), pos)

    # ---------------- 3D flight path viewport ----------------
    def draw_3d(self, screen, sim):
        sub = screen.subsurface(VIEW3D_RECT)
        s = sim.s
        xe_now, ye_now, ze_now = s[10], s[11], s[12]
        alt_now = -ze_now

        # Sky gradient: deep space black at top -> rich cobalt -> horizon haze/amber glow
        haze = max(0.0, min(1.0, 1.0 - alt_now / 40000.0))
        space_frac = max(0.0, min(1.0, alt_now / 80000.0))  # how "space-like" the top looks
        # top colour shifts from near-black space blue to rich night blue
        top_col = (int(2 + 8 * (1 - space_frac)), int(4 + 10 * (1 - space_frac)), int(18 + 30 * (1 - space_frac)))
        # mid colour: rich cobalt blue
        mid_col = (int(8 + 20 * haze), int(30 + 60 * haze), int(100 + 80 * haze))
        # horizon: warm amber/orange haze when low, pale blue when high
        bot_col = (int(60 + 120 * haze * (1 - space_frac)), int(80 + 80 * haze * (1 - space_frac)), int(90 + 60 * haze * (1 - space_frac)))
        bands = 32
        bh = VIEW3D_RECT.height / bands
        for i in range(bands):
            f = i / (bands - 1)
            if f < 0.5:
                # top half: space -> cobalt
                ff = f * 2.0
                col = tuple(int(top_col[k] + (mid_col[k] - top_col[k]) * ff) for k in range(3))
            else:
                # bottom half: cobalt -> horizon
                ff = (f - 0.5) * 2.0
                col = tuple(int(mid_col[k] + (bot_col[k] - mid_col[k]) * ff) for k in range(3))
            pygame.draw.rect(sub, col, (0, i * bh, VIEW3D_RECT.width, bh + 1))

        self.glow3d.clear()

        # ---- filled checkerboard ground plane + wireframe grid ----
        ground_dy = ze_now - 0.0
        span, step = 6000, 1000
        for gx in range(-span, span, step):
            for gz in range(-span, span, step):
                corners = [self._project(gx, ground_dy, gz), self._project(gx + step, ground_dy, gz),
                           self._project(gx + step, ground_dy, gz + step), self._project(gx, ground_dy, gz + step)]
                if all(c is not None for c in corners):
                    checker = ((gx // step) + (gz // step)) % 2
                    col = (18, 52, 22) if checker == 0 else (28, 72, 35)
                    pygame.draw.polygon(sub, col, [c[:2] for c in corners])
        for k in range(-span, span + 1, step):
            p1, p2 = self._project(k, ground_dy, -span), self._project(k, ground_dy, span)
            if p1 and p2:
                pygame.draw.aaline(sub, GRID_COLOR, p1[:2], p2[:2])
            p1, p2 = self._project(-span, ground_dy, k), self._project(span, ground_dy, k)
            if p1 and p2:
                pygame.draw.aaline(sub, GRID_COLOR, p1[:2], p2[:2])

        # trailing ribbon, colored by speed (Violet -> Teal -> Green -> Yellow)
        pts = []
        for (xe, ye, ze, Vt) in sim.trail:
            dx, dz = xe - xe_now, ye - ye_now
            dy = ze_now - ze
            proj = self._project(dx, dy, dz)
            pts.append((proj, Vt))
        for i in range(1, len(pts)):
            (p0, v0), (p1, v1) = pts[i - 1], pts[i]
            if p0 is None or p1 is None:
                continue
            t = (v1 - 150.0) / (750.0)
            col = speed_color(t)
            self.glow3d.line(col, p0[:2], p1[:2], width=2, glow=3, glow_alpha=60)

        R = quat_to_R(s[6], s[7], s[8], s[9])

        def body_to_screen(bx, by, bz):
            ev = R @ np.array([bx, by, bz])
            dx, dz = ev[0], ev[1]
            dy = -ev[2]
            return self._project(dx, dy, dz)

        if not sim.crashed:
            # ================================================================
            # F-16 3D polygon mesh — body-frame vertices, projected to screen.
            # Coordinate system: X = nose, Y = right wing, Z = up (nose up).
            # All units in "display feet" (scaled for visual clarity).
            # ================================================================

            # --- Fuselage cross-sections along X axis ---
            # Each station: (x, half_width_y, half_height_z)
            fuse_stations = [
                (28,  0.0,  0.0),   # nose tip
                (22,  0.8,  0.8),   # nose cone
                (16,  1.4,  1.3),   # forward fuselage
                ( 8,  1.8,  1.6),   # inlet start
                ( 2,  1.8,  1.8),   # widest point
                (-4,  1.6,  1.9),   # mid fuselage
                (-10, 1.5,  1.7),   # aft fuselage
                (-18, 1.2,  1.5),   # tail section
                (-24, 1.0,  1.2),   # nozzle
                (-27, 0.8,  1.0),   # nozzle exit
            ]

            # Generate 8-sided fuselage polygon rings
            N_RING = 8
            def ring_pts(x, wy, wz):
                pts = []
                for k in range(N_RING):
                    ang = 2 * math.pi * k / N_RING
                    py = wy * math.cos(ang)
                    pz = -wz * math.sin(ang)  # z up in body = -z in our convention
                    pts.append((x, py, pz))
                return pts

            rings = [ring_pts(*st) for st in fuse_stations]

            # Collect polygon faces: each face = list of body-frame (x,y,z) verts
            faces = []  # (verts_body, base_color, is_lit)

            # Fuselage panels between adjacent rings
            for ri in range(len(rings) - 1):
                r0, r1 = rings[ri], rings[ri + 1]
                for k in range(N_RING):
                    k2 = (k + 1) % N_RING
                    quad = [r0[k], r0[k2], r1[k2], r1[k]]
                    # shade by Z component (top = lighter, bottom = darker)
                    avg_z = sum(v[2] for v in quad) / 4.0
                    shade = int(120 + 60 * avg_z / 2.0)
                    shade = max(60, min(200, shade))
                    col = (shade - 10, shade + 5, shade + 15)  # slightly blue-grey
                    faces.append((quad, col, True))

            # --- Delta Wings (cranked leading edge, F-16 style) ---
            # Right wing
            rw_verts = [
                ( 8,  1.8,  0.0),   # LERX root leading
                ( 4,  4.0,  0.1),   # LERX outer
                (-2,  7.0,  0.2),   # inner wing LE
                (-8, 14.5,  0.3),   # wingtip LE
                (-13, 14.0,  0.2),  # wingtip TE
                (-14,  4.0,  0.1),  # wing root TE
                (-10,  1.8,  0.0),  # fuselage TE
            ]
            lw_verts = [(x, -y, z) for x, y, z in rw_verts]
            faces.append((rw_verts, (80, 110, 90), True))   # right wing top
            faces.append((lw_verts, (80, 110, 90), True))   # left wing top
            # Wing undersides (slightly darker)
            rw_bot = [(x, y, z - 0.3) for x, y, z in rw_verts]
            lw_bot = [(x, -y, z - 0.3) for x, y, z in rw_verts]
            faces.append((list(reversed(rw_bot)), (55, 80, 65), True))
            faces.append((list(reversed(lw_bot)), (55, 80, 65), True))

            # --- Horizontal Stabilators ---
            rs_verts = [
                (-16,  1.5,  0.5),
                (-17,  7.5,  0.6),
                (-22,  7.0,  0.5),
                (-22,  1.5,  0.4),
            ]
            ls_verts = [(x, -y, z) for x, y, z in rs_verts]
            faces.append((rs_verts, (70, 100, 80), True))
            faces.append((ls_verts, (70, 100, 80), True))

            # --- Vertical Tail ---
            vt_verts = [
                (-10,  0.0, -1.5),   # root leading edge
                (-10,  0.0, -9.5),   # tip leading edge
                (-22,  0.0, -9.0),   # tip trailing edge
                (-22,  0.0, -1.5),   # root trailing edge
            ]
            # Left & right faces of fin
            vt_r = [(x,  0.35, z) for x, y, z in vt_verts]
            vt_l = [(x, -0.35, z) for x, y, z in vt_verts]
            faces.append((vt_r, (90, 120, 100), True))
            faces.append((list(reversed(vt_l)), (90, 120, 100), True))

            # --- Intake cheeks (rectangular boxes under nose) ---
            intake_verts_r = [
                ( 6,  1.0, 0.8),
                ( 6,  3.2, 0.8),
                (-2,  3.2, 1.0),
                (-2,  1.0, 1.0),
            ]
            intake_verts_l = [(x, -y, z) for x, y, z in intake_verts_r]
            faces.append((intake_verts_r, (40, 60, 50), True))
            faces.append((intake_verts_l, (40, 60, 50), True))

            # --- Canopy ---
            canopy_verts = [
                (14,  1.0, -1.2),
                (14, -1.0, -1.2),
                ( 6, -1.2, -2.5),
                ( 4, -1.0, -2.6),
                ( 4,  1.0, -2.6),
                ( 6,  1.2, -2.5),
            ]
            faces.append((canopy_verts, (60, 160, 220), True))  # tinted blue glass

            # --- Project all faces and depth-sort (painter's algorithm) ---
            def project_face(face_verts):
                screen_pts = []
                depths = []
                for bx, by, bz in face_verts:
                    sp = body_to_screen(bx, by, bz)
                    if sp is None:
                        return None, None
                    screen_pts.append(sp[:2])
                    depths.append(sp[2])
                return screen_pts, sum(depths) / len(depths)

            # Compute face normals for backface culling + simple lighting
            def face_normal_and_center(verts):
                if len(verts) < 3:
                    return None, None
                # Use first 3 verts for normal
                v0 = np.array(verts[0])
                v1 = np.array(verts[1])
                v2 = np.array(verts[2])
                n = np.cross(v1 - v0, v2 - v0)
                nlen = np.linalg.norm(n)
                if nlen < 1e-9:
                    return None, None
                return n / nlen, v0

            # Camera view direction in body frame (approximate: use R^T * cam_dir)
            cy_cam = math.cos(self.cam_yaw)
            sy_cam = math.sin(self.cam_yaw)
            cp_cam = math.cos(self.cam_pitch)
            sp_cam = math.sin(self.cam_pitch)
            # world-to-body: R^T (rotation matrix from quat)
            R_mat = quat_to_R(s[6], s[7], s[8], s[9])
            # camera forward in world frame
            cam_world = np.array([-sy_cam * cp_cam, -sp_cam, -cy_cam * cp_cam])
            cam_body = R_mat.T @ cam_world

            # Sun direction for diffuse lighting (fixed world direction)
            sun_world = np.array([0.4, 0.3, -0.85])
            sun_world /= np.linalg.norm(sun_world)
            sun_body = R_mat.T @ sun_world

            # Build list of (depth, screen_pts, shaded_color)
            draw_list = []
            for (verts, base_col, is_lit) in faces:
                norm, _ = face_normal_and_center(verts)
                if norm is not None:
                    # Backface cull
                    if np.dot(norm, cam_body) > 0.05:
                        continue
                    if is_lit:
                        diffuse = max(0.0, -np.dot(norm, sun_body))
                        ambient = 0.35
                        light = ambient + (1.0 - ambient) * diffuse
                        shaded = tuple(min(255, int(c * light)) for c in base_col)
                    else:
                        shaded = base_col
                else:
                    shaded = base_col

                screen_pts, depth = project_face(verts)
                if screen_pts is None:
                    continue
                draw_list.append((depth, screen_pts, shaded))

            # Painter's algorithm: draw back-to-front
            draw_list.sort(key=lambda x: -x[0])
            for depth, pts, col in draw_list:
                if len(pts) >= 3:
                    try:
                        pygame.draw.polygon(sub, col, [(int(p[0]), int(p[1])) for p in pts])
                        # Thin edge outline for definition
                        pygame.draw.polygon(sub, tuple(min(255, c + 30) for c in col),
                                            [(int(p[0]), int(p[1])) for p in pts], 1)
                    except Exception:
                        pass

            # Cockpit highlight line
            cap_pts, _ = project_face(canopy_verts)
            if cap_pts and len(cap_pts) >= 3:
                pygame.draw.polygon(self.glow3d.surf,
                                    (100, 200, 255, 120),
                                    [(int(p[0]), int(p[1])) for p in cap_pts])

            # ---- afterburner plume (flickers/lengthens with throttle) ----
            throttle = sim.last_ctrl_out[3]
            nozzle = body_to_screen(-27, 0, 1)
            if throttle > 65 and nozzle:
                ab_frac = max(0.0, min(1.0, (throttle - 65) / 35.0))
                flicker = 0.85 + 0.15 * math.sin(sim.t * 40.0) + self._rng_jitter()
                length = 10 + 28 * ab_frac * flicker
                for seg in range(5):
                    f0, f1 = seg / 5.0, (seg + 1) / 5.0
                    p_a = body_to_screen(-27 - length * f0, 0, 1)
                    p_b = body_to_screen(-27 - length * f1, 0, 1)
                    if p_a and p_b:
                        col = (255, 240, 180) if seg == 0 else (255, 160, 40) if seg < 3 else (255, 50, 10)
                        w = max(1, 6 - seg)
                        self.glow3d.line(col, p_a[:2], p_b[:2], width=w, glow=7 - seg, glow_alpha=120)

        # ---- explosion particles ----
        for p in sim.explosion_particles:
            proj = self._project(p[0], -p[2], p[1])
            if proj:
                life_frac = max(0.0, p[6] / p[7])
                col = speed_color(1.0) if life_frac > 0.6 else (255, int(140 * life_frac), 20)
                r = max(1, int(6 * life_frac))
                self.glow3d.circle(col, (int(proj[0]), int(proj[1])), r, glow=int(3 + 4 * life_frac), glow_alpha=140)

        self.glow3d.blit_to(sub, (0, 0))

        if sim.crash_flash > 0.0:
            flash = pygame.Surface(VIEW3D_RECT.size, pygame.SRCALPHA)
            flash.fill((255, 160, 60, int(160 * sim.crash_flash)))
            sub.blit(flash, (0, 0))

        pygame.draw.rect(sub, BORDER, sub.get_rect(), width=1)
        self.draw_text(sub, "3D FLIGHT PATH  (drag = rotate, scroll = zoom)", (10, 8), self.font_sm, DIM_TEXT)
        self.draw_text(sub, f"ALT {alt_now:8.0f} ft", (10, 26), self.font_sm, TEXT_COLOR)

        if sim.crashed:
            msg = self.font_xl.render("CRASHED", True, RED)
            sub_msg = self.font_md.render("press R to respawn", True, YELLOW)
            sub.blit(msg, (VIEW3D_RECT.width / 2 - msg.get_width() / 2, VIEW3D_RECT.height / 2 - 40))
            sub.blit(sub_msg, (VIEW3D_RECT.width / 2 - sub_msg.get_width() / 2, VIEW3D_RECT.height / 2 + 6))

    def _rng_jitter(self):
        return (np.random.random() - 0.5) * 0.15

    # ---------------- phase-space viewport ----------------
    def draw_phase(self, screen, sim):
        sub = screen.subsurface(PHASE_RECT)
        sub.fill(BG_COLOR)
        self.glow_phase.clear()

        aux = sim.last_aux
        alpha_deg, beta_deg = aux[0], aux[1]
        q_dps = math.degrees(sim.s[4])
        p_dps = math.degrees(sim.s[3])
        self.phase_hist.append((float(alpha_deg), float(q_dps), float(beta_deg), float(p_dps)))
        if len(self.phase_hist) > self.phase_hist_max:
            self.phase_hist.pop(0)

        w, h = PHASE_RECT.width, PHASE_RECT.height
        plot_w, plot_h = w - 40, (h - 60) // 2

        def mini_plot(y0, xlabel, ylabel, xrange, yrange, series, idx_x, idx_y, color):
            x0 = 34
            rect = pygame.Rect(x0, y0, plot_w, plot_h)
            pygame.draw.rect(sub, (12, 16, 20), rect)
            pygame.draw.rect(sub, GRID_COLOR, rect, width=1)
            for gx in range(5):
                fx = gx / 4.0
                xx = rect.left + fx * rect.width
                pygame.draw.line(sub, GRID_COLOR, (xx, rect.top), (xx, rect.bottom))
                val = xrange[0] + fx * (xrange[1] - xrange[0])
                self.draw_text(sub, f"{val:.0f}", (xx - 8, rect.bottom + 2), self.font_sm, DIM_TEXT)
            for gy in range(5):
                fy = gy / 4.0
                yy = rect.top + fy * rect.height
                pygame.draw.line(sub, GRID_COLOR, (rect.left, yy), (rect.right, yy))
                val = yrange[1] - fy * (yrange[1] - yrange[0])
                self.draw_text(sub, f"{val:5.0f}", (2, yy - 6), self.font_sm, DIM_TEXT)
            self.draw_text(sub, ylabel, (rect.left + 4, rect.top - 14), self.font_sm, color)
            self.draw_text(sub, xlabel, (rect.right - 40, rect.bottom + 14), self.font_sm, color)

            n = len(series)
            prev = None
            for i, sample in enumerate(series):
                xv, yv = sample[idx_x], sample[idx_y]
                fx = (xv - xrange[0]) / (xrange[1] - xrange[0])
                fy = (yv - yrange[0]) / (yrange[1] - yrange[0])
                fx, fy = max(0, min(1, fx)), max(0, min(1, fy))
                px = rect.left + fx * rect.width
                py = rect.bottom - fy * rect.height
                fade = i / max(1, n - 1)
                if prev is not None:
                    a = int(40 + 200 * fade)
                    r, g, b = color
                    pygame.draw.line(self.glow_phase.surf, (r, g, b, a), prev, (px, py), 1)
                prev = (px, py)
            if prev:
                self.glow_phase.circle(color, (int(prev[0]), int(prev[1])), 4, glow=6, glow_alpha=100)

        mini_plot(28, "alpha (deg)", "q (deg/s)", (-10, 45), (-100, 100),
                  self.phase_hist, 0, 1, TEAL)
        mini_plot(28 + plot_h + 34, "beta (deg)", "p (deg/s)", (-30, 30), (-250, 250),
                  self.phase_hist, 2, 3, VIOLET)

        self.glow_phase.blit_to(sub, (0, 0))
        pygame.draw.rect(sub, BORDER, sub.get_rect(), width=1)
        self.draw_text(sub, "PHASE-SPACE  (q vs alpha  /  p vs beta)", (10, 6), self.font_sm, DIM_TEXT)

    # ---------------- HUD viewport ----------------
    @staticmethod
    def _rot(x, y, ang):
        c, s = math.cos(ang), math.sin(ang)
        return x * c - y * s, x * s + y * c

    def draw_pitch_ladder(self, sub, rect, theta, phi):
        pygame.draw.rect(sub, (6, 12, 12), rect)
        cx, cy = rect.center
        ppd = rect.height / 60.0
        theta_deg = math.degrees(theta)
        for pv in range(-90, 91, 10):
            local_y = (theta_deg - pv) * ppd
            if abs(local_y) > rect.height / 2 + 20:
                continue
            half_w = 85 if pv == 0 else 50
            gap = 0 if pv == 0 else 16
            color = TEAL if pv == 0 else (NEON_GREEN if pv > 0 else AMBER)
            segs = [(-half_w, local_y, -gap, local_y), (gap, local_y, half_w, local_y)]
            for (x1, y1, x2, y2) in segs:
                rx1, ry1 = self._rot(x1, y1, -phi)
                rx2, ry2 = self._rot(x2, y2, -phi)
                pygame.draw.aaline(sub, color, (cx + rx1, cy + ry1), (cx + rx2, cy + ry2))
            if pv != 0:
                lx, ly = self._rot(half_w + 16, local_y, -phi)
                self.draw_text(sub, f"{pv:d}", (cx + lx - 9, cy + ly - 6), self.font_sm, color)
        # fixed boresight / velocity-vector marker
        pygame.draw.circle(sub, YELLOW, (cx, cy), 4, width=2)
        pygame.draw.line(sub, YELLOW, (cx - 16, cy), (cx - 6, cy), 2)
        pygame.draw.line(sub, YELLOW, (cx + 6, cy), (cx + 16, cy), 2)
        pygame.draw.line(sub, YELLOW, (cx, cy - 12), (cx, cy - 6), 2)
        pygame.draw.rect(sub, BORDER, rect, width=1)

    def draw_compass(self, sub, rect, psi_deg):
        pygame.draw.rect(sub, (6, 12, 12), rect)
        pygame.draw.rect(sub, BORDER, rect, width=1)
        cx = rect.centerx
        span_deg = 60
        px_per_deg = rect.width / span_deg
        start = int(psi_deg - span_deg / 2)
        end = int(psi_deg + span_deg / 2)
        for h in range(start - (start % 10), end + 10, 10):
            dx = (h - psi_deg) * px_per_deg
            x = cx + dx
            if rect.left < x < rect.right:
                hh = ((h % 360) + 360) % 360
                pygame.draw.line(sub, TEAL, (x, rect.top + 4), (x, rect.top + 14), 2)
                self.draw_text(sub, f"{hh:03d}", (x - 12, rect.top + 16), self.font_sm, TEAL)
        pygame.draw.polygon(sub, YELLOW, [(cx, rect.top + 2), (cx - 6, rect.top - 6), (cx + 6, rect.top - 6)])

    def draw_gmeter(self, sub, rect, nz):
        pygame.draw.rect(sub, (6, 12, 12), rect)
        pygame.draw.rect(sub, BORDER, rect, width=1)
        gmin, gmax = -3.0, 9.0
        f = (nz - gmin) / (gmax - gmin)
        f = max(0.0, min(1.0, f))
        bar_h = int(f * (rect.height - 6))
        col = RED if (nz > 7.5 or nz < -2.5) else NEON_GREEN
        pygame.draw.rect(sub, col, (rect.left + 3, rect.bottom - 3 - bar_h, rect.width - 6, bar_h))
        self.draw_text(sub, f"{nz:4.1f}G", (rect.left + 2, rect.top - 16), self.font_sm, TEXT_COLOR)

    def draw_bar(self, sub, rect, val, vmax, label, color):
        pygame.draw.rect(sub, (6, 12, 12), rect)
        pygame.draw.rect(sub, GRID_COLOR, rect, width=1)
        cx = rect.centerx
        pygame.draw.line(sub, DIM_TEXT, (cx, rect.top), (cx, rect.bottom), 1)
        f = max(-1.0, min(1.0, val / vmax))
        w = int(abs(f) * (rect.width / 2 - 3))
        if f >= 0:
            pygame.draw.rect(sub, color, (cx, rect.top + 3, w, rect.height - 6))
        else:
            pygame.draw.rect(sub, color, (cx - w, rect.top + 3, w, rect.height - 6))
        self.draw_text(sub, f"{label} {val:+5.1f}", (rect.left, rect.bottom + 2), self.font_sm, DIM_TEXT)

    def draw_hud(self, screen, sim):
        sub = screen.subsurface(HUD_RECT)
        sub.fill(PANEL_BG)
        s = sim.s
        aux = sim.last_aux
        alpha_deg, beta_deg, Nz, Ny, Vt, mach, qbar, T = aux
        q0, q1, q2, q3 = s[6], s[7], s[8], s[9]
        phi = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        s_theta = max(-1.0, min(1.0, 2 * (q0 * q2 - q3 * q1)))
        theta = math.asin(s_theta)
        psi = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        psi_deg = (math.degrees(psi) + 360) % 360
        alt = -s[12]

        self.draw_text(sub, "FLIGHT HUD", (10, 6), self.font_sm, DIM_TEXT)

        ladder_rect = pygame.Rect(12, 26, 300, 190)
        self.draw_pitch_ladder(sub, ladder_rect, theta, phi)

        compass_rect = pygame.Rect(12, 224, 300, 40)
        self.draw_compass(sub, compass_rect, psi_deg)

        gm_rect = pygame.Rect(330, 26, 26, 190)
        self.draw_gmeter(sub, gm_rect, Nz)

        rx = 372
        readouts = [
            ("ALT", f"{alt:8.0f} ft", TEAL),
            ("SPD", f"{Vt*0.5925:8.0f} kt", TEAL),
            ("MACH", f"{mach:8.2f}", TEAL),
            ("AOA", f"{alpha_deg:8.1f} dg", AMBER),
            ("SLIP", f"{beta_deg:8.1f} dg", AMBER),
            ("THR", f"{sim.last_ctrl_out[3]:8.0f} %", NEON_GREEN),
        ]
        for i, (label, val, col) in enumerate(readouts):
            y = 30 + i * 30
            self.draw_text(sub, label, (rx, y), self.font_sm, DIM_TEXT)
            self.draw_text(sub, val, (rx, y + 14), self.font_md, col)

        el, ail, rud, thr = sim.last_ctrl_out
        bar_y = 232
        self.draw_bar(sub, pygame.Rect(372, bar_y, 200, 14), el, 25, "EL", TEAL)
        self.draw_bar(sub, pygame.Rect(372, bar_y + 40, 200, 14), ail, 21.5, "AIL", VIOLET)
        self.draw_bar(sub, pygame.Rect(372, bar_y + 80, 200, 14), rud, 30, "RUD", AMBER)

        status_y = HUD_RECT.height - 26
        self.draw_text(sub, f">> {sim.autopilot.status_text}", (12, status_y), self.font_md, YELLOW)
        pygame.draw.rect(sub, BORDER, sub.get_rect(), width=1)

    # ---------------- control panel ----------------
    def draw_panel(self, screen, sim):
        sub = screen.subsurface(PANEL_RECT)
        sub.fill(PANEL_BG)
        self.draw_text(sub, "MANEUVERS", (10, 6), self.font_sm, DIM_TEXT)
        self.buttons = []
        cols = 4
        margin = 10
        bw = (PANEL_RECT.width - margin * (cols + 1)) // cols
        bh = 62
        for i, name in enumerate(MANEUVERS):
            col = i % cols
            row = i // cols
            x = margin + col * (bw + margin)
            y = 24 + row * (bh + margin)
            rect = pygame.Rect(x, y, bw, bh)
            active = (sim.autopilot.mode == name)
            base_col = YELLOW if active else BORDER
            glow_col = tuple(min(255, c + 40) for c in base_col) if active else base_col
            pygame.draw.rect(sub, (10, 18, 18) if not active else (25, 25, 10), rect, border_radius=8)
            pygame.draw.rect(sub, glow_col, rect, width=2, border_radius=8)
            lines = [name] if len(name) <= 12 else [name[:len(name) // 2].strip(), name[len(name) // 2:].strip()]
            for li, line in enumerate(lines):
                txt = self.font_sm.render(line, True, base_col if not active else YELLOW)
                tw, th = txt.get_size()
                sub.blit(txt, (rect.centerx - tw / 2, rect.centery - (len(lines) * th) / 2 + li * th))
            # translate button rect into full-screen coords for click testing
            self.buttons.append((rect.move(PANEL_RECT.topleft), name))
        pygame.draw.rect(sub, BORDER, sub.get_rect(), width=1)

    def draw(self, screen, sim):
        screen.fill(BG_COLOR)
        self.draw_3d(screen, sim)
        self.draw_phase(screen, sim)
        self.draw_hud(screen, sim)
        self.draw_panel(screen, sim)


# ==============================================================================
# 12. PROCEDURAL SOUND (no external audio files / no copyrighted clips —
#     everything below is synthesized on the fly with numpy)
# ==============================================================================
class SoundEngine:
    def __init__(self):
        self.enabled = True
        self.sr = 44100
        try:
            pygame.mixer.init(frequency=self.sr, size=-16, channels=2, buffer=512)
            self.ch_engine = pygame.mixer.Channel(0)
            self.ch_ab = pygame.mixer.Channel(1)
            self.ch_wind = pygame.mixer.Channel(2)
            self.ch_fx = pygame.mixer.Channel(3)
            self._engine_band = -1
            self._ab_playing = False
            self.wind_sound = self._noise_loop(0.6, lowpass=0.15)
            self.explosion_sound = self._explosion_sound()
            self.click_sound = self._click_sound()
            self.ab_sound = self._noise_loop(0.4, lowpass=0.35, amp=0.9)
        except Exception as e:
            print(f"[sound] audio unavailable, running muted ({e})")
            self.enabled = False

    def _to_sound(self, wave):
        wave = np.clip(wave, -1.0, 1.0)
        mono = (wave * 32767 * 0.6).astype(np.int16)
        stereo = np.column_stack([mono, mono])
        return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))

    def _tone_loop(self, f0, duration, sr):
        n = int(sr * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        wave = (0.9 * np.sin(2 * np.pi * f0 * t) + 0.45 * np.sin(2 * np.pi * f0 * 2 * t)
                + 0.25 * np.sin(2 * np.pi * f0 * 3 * t) + 0.5 * np.sin(2 * np.pi * f0 * 0.5 * t))
        wave /= np.max(np.abs(wave)) + 1e-9
        return wave

    def _engine_band_sound(self, band):
        f0 = 55 + band * 30
        # choose a duration that's a whole number of fundamental periods -> seamless loop
        periods = max(1, round(0.35 * f0))
        duration = periods / f0
        wave = self._tone_loop(f0, duration, self.sr)
        return self._to_sound(wave)

    def _noise_loop(self, duration, lowpass=0.2, amp=0.7):
        n = int(self.sr * duration)
        white = np.random.uniform(-1, 1, n)
        # crude one-pole low-pass filter for a dull rumble instead of hiss
        filt = np.zeros(n)
        acc = 0.0
        for i in range(n):
            acc = acc + lowpass * (white[i] - acc)
            filt[i] = acc
        filt /= np.max(np.abs(filt)) + 1e-9
        # fade the loop edges so the wraparound doesn't click
        fade = min(400, n // 8)
        env = np.ones(n)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        return self._to_sound(filt * env * amp)

    def _explosion_sound(self):
        dur = 1.1
        n = int(self.sr * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        noise = np.random.uniform(-1, 1, n)
        env = np.exp(-t * 3.2)
        rumble = 0.6 * np.sin(2 * np.pi * 55 * t) * np.exp(-t * 2.0)
        wave = (noise * env + rumble)
        return self._to_sound(wave)

    def _click_sound(self):
        dur = 0.06
        n = int(self.sr * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        wave = np.sin(2 * np.pi * 1200 * t) * np.exp(-t * 60)
        return self._to_sound(wave)

    def update(self, throttle_pct, vt, dt):
        if not self.enabled:
            return
        band = int(np.clip(throttle_pct / 100.0 * 4, 0, 4))
        if band != self._engine_band:
            self.ch_engine.play(self._engine_band_sound(band), loops=-1, fade_ms=200)
            self._engine_band = band
        self.ch_engine.set_volume(min(1.0, 0.25 + 0.55 * (throttle_pct / 100.0)))

        ab_on = throttle_pct > 70.0
        if ab_on and not self._ab_playing:
            self.ch_ab.play(self.ab_sound, loops=-1, fade_ms=120)
            self._ab_playing = True
        elif not ab_on and self._ab_playing:
            self.ch_ab.fadeout(200)
            self._ab_playing = False
        if ab_on:
            self.ch_ab.set_volume(float(np.clip((throttle_pct - 70) / 30.0, 0, 1)) * 0.8)

        if not self.ch_wind.get_busy():
            self.ch_wind.play(self.wind_sound, loops=-1)
        self.ch_wind.set_volume(float(np.clip((vt - 100) / 700.0, 0, 1)) * 0.3)

    def play_explosion(self):
        if self.enabled:
            self.ch_fx.play(self.explosion_sound)

    def play_click(self):
        if self.enabled:
            self.ch_fx.play(self.click_sound)


# ==============================================================================
# 13. MAIN
# ==============================================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("F-16 Flight Dynamics Simulator")
    clock = pygame.time.Clock()

    sim = F16Simulation()
    vis = Visualizer()
    sound = SoundEngine()

    physics_dt = sim.dt
    accumulator = 0.0
    running = True
    was_crashed = False

    max_substeps_per_frame = 12  # safety valve if the machine can't keep up

    while running:
        frame_dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if sim.crashed:
                    sim.reset()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                label = vis.button_clicked(event.pos)
                if label is not None:
                    if not sim.crashed:
                        sim.autopilot.start(label)
                        sound.play_click()
                else:
                    vis.handle_event(event)
            else:
                vis.handle_event(event)

        accumulator += frame_dt
        n = 0
        while accumulator >= physics_dt and n < max_substeps_per_frame:
            sim.step()
            accumulator -= physics_dt
            n += 1

        if sim.crashed and not was_crashed:
            sound.play_explosion()
        was_crashed = sim.crashed
        sound.update(sim.last_ctrl_out[3], sim.last_aux[4] if sim.last_aux is not None else 0.0, frame_dt)

        vis.draw(screen, sim)
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
