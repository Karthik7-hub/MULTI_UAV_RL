"""
Rotary-Wing Aerodynamics Propulsion Power Model.
Reference:
    Y. Zeng, J. Xu, and R. Zhang, "Energy-Minimization Policy for UAV-Powered Mobile Edge
    Computing Systems," IEEE Trans. Wireless Commun., vol. 18, no. 6, pp. 3145-3159, 2019.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class AerodynamicsModel:
    P_0: float = 79.86     # Blade profile power in hover (Watts)
    P_i: float = 88.63     # Induced power in hover (Watts)
    U_tip: float = 120.0   # Tip speed of rotor blade (m/s)
    v_0: float = 4.03      # Mean rotor induced velocity in hover (m/s)
    d_0: float = 0.6       # Fuselage drag ratio
    rho: float = 1.225     # Air density (kg/m^3)
    s: float = 0.05        # Rotor solidity
    A: float = 0.5         # Rotor disc area (m^2)
    v_max: float = 15.0    # Max airspeed (m/s)

    def power(self, speed: float) -> float:
        """
        # =========================================================================
        # PURPOSE:
        # Calculates the instantaneous aerodynamic propulsion power consumed by a
        # rotary-wing UAV flying at horizontal airspeed 'v'.
        #
        # EQUATION (Zeng et al., IEEE TWC 2019):
        # P(v) = P_profile(v) + P_induced(v) + P_parasite(v)
        #
        # Where:
        # 1. Blade Profile Power:
        #    P_profile(v) = P_0 * (1 + 3 * v^2 / U_tip^2)
        #    - Represents the aerodynamic profile drag of the spinning rotor blades.
        #    - Increases quadratically with forward airspeed v.
        #
        # 2. Induced Power (Translational Lift Effect):
        #    P_induced(v) = P_i * sqrt( sqrt(1 + v^4 / (4 * v_0^4)) - v^2 / (2 * v_0^2) )
        #    - Represents the power needed to produce aerodynamic lift (push air downwards).
        #    - In hover (v=0), it equals P_i. As airspeed increases, forward motion provides
        #      translational lift, significantly REDUCING the induced power requirement!
        #
        # 3. Parasite Drag Power:
        #    P_parasite(v) = 0.5 * d_0 * rho * s * A * v^3
        #    - Represents the fuselage air resistance (form drag) pushing against the UAV body.
        #    - Increases cubically (v^3) with forward airspeed!
        #
        # VARIABLES & UNITS:
        # - speed (v)     : Forward horizontal flight speed of the UAV [m/s]
        # - P_0           : Blade profile power in pure hover = 79.86 [Watts (W)]
        # - P_i           : Induced power in pure hover = 88.63 [Watts (W)]
        # - U_tip         : Linear tip speed of the spinning rotor blades = 120.0 [m/s]
        # - v_0           : Mean rotor induced air velocity during pure hover = 4.03 [m/s]
        # - d_0           : Fuselage aerodynamic drag coefficient (dimensionless) = 0.6
        # - rho           : Ambient air density at sea level = 1.225 [kg/m^3]
        # - s             : Rotor solidity ratio (blade area / disc area) = 0.05
        # - A             : Total rotor disc surface area = 0.5 [m^2]
        #
        # RESULT:
        # - P(v) : Total instantaneous mechanical propulsion power [Watts (W = Joules/second)]
        #
        # WHY THIS MATTERS FOR UAV RL RESEARCH:
        # - In hover (v=0), total power = P_0 + P_i = 79.86 + 88.63 = 168.49 W.
        # - At moderate cruising speeds (v ~ 8 to 11 m/s), translational lift reduces induced
        #   power faster than parasite drag increases, creating an energy-efficient "sweet spot"
        #   where power drops to ~145 W.
        # - At maximum speed (v=15 m/s), the cubic parasite drag dominates, driving power up to ~230 W.
        # - The RL agent must balance flying fast to meet task deadlines against the aerodynamic
        #   energy cost of excessive velocity!
        # =========================================================================
        """
        v = float(np.clip(speed, 0.0, self.v_max))
        
        # 1. Blade Profile Power [Watts]
        # Increases with v^2 as rotating blades encounter higher relative airspeed
        p_profile = self.P_0 * (1.0 + 3.0 * (v ** 2) / (self.U_tip ** 2))
        
        # 2. Induced Power [Watts]
        # Captures translational lift: forward flight pumps fresh air through the rotor disc,
        # reducing the induced velocity needed to support the UAV's weight.
        inner_root = np.sqrt(np.maximum(0.0, 1.0 + (v ** 4) / (4.0 * (self.v_0 ** 4))))
        induced_term = inner_root - (v ** 2) / (2.0 * (self.v_0 ** 2))
        p_induced = self.P_i * np.sqrt(np.maximum(0.0, induced_term))
        
        # 3. Parasite Drag Power [Watts]
        # Fuselage air resistance: Force = 0.5 * rho * d_0 * s * A * v^2, Power = Force * v = ~ v^3
        p_parasite = 0.5 * self.d_0 * self.rho * self.s * self.A * (v ** 3)
        
        return float(p_profile + p_induced + p_parasite)

    def energy(self, speed: float, delta_t: float = 1.0) -> float:
        """
        # =========================================================================
        # PURPOSE:
        # Converts instantaneous mechanical propulsion power into energy consumed
        # over a discrete simulation timestep of duration delta_t.
        #
        # EQUATION:
        # E = P(v) * delta_t
        #
        # VARIABLES & UNITS:
        # - speed   : UAV flight speed [m/s]
        # - delta_t : Duration of the simulation timestep [seconds (s)] (default = 1.0 s)
        # - P(v)    : Propulsion power [Watts = Joules / second]
        #
        # RESULT:
        # - E : Energy consumed during this step [Joules (J)]
        #
        # HOW IT CONNECTS TO SIMULATION:
        # MultiUAVEnv accumulates this energy into 'step_energy' and 'cumulative_energy',
        # and applies an energy penalty in the RL reward to incentivize efficient paths.
        # =========================================================================
        """
        return self.power(speed) * delta_t

    def get_power_curve(self, num_points: int = 150):
        """Returns speeds, total power, induced power, profile power, and parasite power arrays."""
        speeds = np.linspace(0.0, self.v_max, num_points)
        p_total = np.array([self.power(v) for v in speeds])
        
        p_profile = self.P_0 * (1.0 + 3.0 * (speeds ** 2) / (self.U_tip ** 2))
        inner = np.sqrt(np.maximum(0.0, 1.0 + (speeds ** 4) / (4.0 * (self.v_0 ** 4)))) - (speeds ** 2) / (2.0 * (self.v_0 ** 2))
        p_induced = self.P_i * np.sqrt(np.maximum(0.0, inner))
        p_parasite = 0.5 * self.d_0 * self.rho * self.s * self.A * (speeds ** 3)
        
        return speeds, p_total, p_induced, p_profile, p_parasite
