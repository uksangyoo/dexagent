"""Declared soft-finger friction law, not a task-dependent friction schedule.

Normal contact remains MuJoCo's compliant convex model. Only torsional
friction uses the Hertz pressure-profile approximation:
    a = (3 N R_eff / (4 E*))**(1/3), T_max/N = 3*pi*mu*a/16.
Shore A is not a calibration of pad modulus, thickness, or friction. Gent's
estimate is a nominal assumption; E and effective curvature remain uncertain.

Primary sources:
https://www.cs.columbia.edu/~cmatei/thesis.pdf (chapter 7)
https://doi.org/10.1177/02783649922066673 (Xydas & Kao, 1999)
https://doi.org/10.1016/j.dental.2009.02.001 (Meththananda et al., 2009)
"""
import numpy as np

SHORE_A=40.
YOUNG_PA=1e6*.0981*(56+7.62336*SHORE_A)/(.137505*(254-2.54*SHORE_A))
POISSON=.49
PAD_RADIUS=.010

def torsion_length(load,mu,radius=.012,young=YOUNG_PA):
    effective_radius=1/(1/PAD_RADIUS+1/radius)
    e_star=young/(1-POISSON**2)
    a=(3*max(0.,load)*effective_radius/(4*e_star))**(1/3)
    return 3*np.pi/16*mu*a

def update(s):
    loads=[s.handleloads[0],s.index_cap_load,s.handleloads[1],s.loads[3],s.loads[4]]
    for i,geom in enumerate(s.tipids):
        # Flat cap uses the pad's curvature alone. No task phase enters this
        # material law. Load is the previous solved contact sample (1 ms).
        radius=1e9 if i==1 else float(s.m.geom('handle').size[0])
        s.m.geom_friction[geom,1]=torsion_length(loads[i],s.m.geom_friction[geom,0],radius,
                                               getattr(s,'pad_young_pa',YOUNG_PA))

def description():
    return dict(model='load-dependent soft-finger torsional friction; compliant convex normal contact',
        young_modulus_Pa=YOUNG_PA,poisson_ratio=POISSON,pad_curvature_radius_m=PAD_RADIUS,
        torsion_law='Tmax=(3*pi/16)*mu*N*(3*N*Reff/(4*Estar))^(1/3)',
        load_update_s=.001,calibrated=False,
        limitations=['Gent hardness-to-modulus estimate, not measured silicone modulus',
                    'Circular Hertz patch approximates curved-pad/cylindrical-handle contact',
                    'Normal compliance is the declared MuJoCo model, not a Hertz indentation solve',
                    'No pad thickness, viscoelastic hysteresis or tactile sensor model'])
