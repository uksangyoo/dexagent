"""Independent checks on measured fastener motion, never motion constraints."""
import math

def valid_thread_motion(rotation_deg,advance_mm,pitch_mm=.5,backlash_tolerance_mm=.1):
    values=[rotation_deg,advance_mm,pitch_mm,backlash_tolerance_mm]
    return all(math.isfinite(x)for x in values)and rotation_deg>10 and advance_mm>0 and abs(advance_mm-rotation_deg*pitch_mm/360)<=backlash_tolerance_mm
