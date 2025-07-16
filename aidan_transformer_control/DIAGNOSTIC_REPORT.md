# CartPole Swing-Up Controller Diagnostic Report

## Executive Summary

The CartPole swing-up controller exhibits **catastrophic instability** when initialized with random initial conditions, despite working perfectly with hanging initial conditions. The root cause is a **mode-switching pathology** that creates uncontrolled oscillations between LQR and swing-up control modes.

## Key Findings

### 1. **Mode Switching Pathology**
- **Hanging start**: 1 mode switch in 200 steps → **SUCCESS**
- **Random start**: 128 mode switches in 500 steps → **FAILURE**
- **Pattern**: Rapid oscillation between LQR ↔ Swing-up every 1-6 steps

### 2. **Energy Explosion**
- **Hanging start**: Energy error mean = -0.318, max = 0.061
- **Random start**: Energy error mean = 4999, max = 20101
- **Target energy**: ~20-30 for typical parameters

### 3. **Force Saturation**
- **Hanging start**: Forces max = 20.588N (reasonable)
- **Random start**: Forces max = 500.0N (saturated at limit)

### 4. **Theta Accumulation Problem**
- System accumulates multiple rotations: `θ=1447.32°`, `θ=5751.94°`, etc.
- LQR controller receives massive unwrapped angles
- Causes immediate mode switching back to swing-up

## Detailed Analysis

### Mode Switching Pattern
```
Step 25: Swing-up → LQR   (θ=-25.32°)
Step 29: LQR → Swing-up   (θ=-29.29°)  ← Immediate switch back
Step 39: Swing-up → LQR   (θ=20.92°)
Step 40: LQR → Swing-up   (θ=-39.24°)  ← Immediate switch back
```

### Energy Control Breakdown
- **Brake reasons distribution**:
  - Hanging: `{'pump_energy': 33, 'coast': 129}` (healthy)
  - Random: `{'pump_energy': 5, 'coast': 119, 'brake_energy': 142}` (excessive braking)

### Parameter Sensitivity
- **Mass pole modifier**: 8.59 (vs 1.0 default)
- **Length modifier**: 2.86 (vs 1.0 default)
- Higher inertia + longer pole = more energy storage = harder control

## Root Cause Analysis

### Primary Issue: **Hysteresis Absence**
The mode switching logic lacks hysteresis:
```python
if abs(theta_upright) < self.switch_angle:  # LQR
else:                                       # Swing-up
```

### Secondary Issues:
1. **Angle wrapping inconsistency** between energy calculation and mode switching
2. **LQR controller instability** with large initial angles
3. **Energy controller over-aggressiveness** with high-energy initial conditions

## Failure Mechanism

1. **High-energy initial condition** (random θ, θ̇)
2. **Swing-up controller** tries to manage energy
3. **Brief approach to upright** triggers LQR mode
4. **LQR gets large angle** → applies large force → destabilizes
5. **Immediate switch back** to swing-up
6. **Repeat cycle** → energy builds up → forces saturate → system diverges

## Recommendations

### Immediate Fixes (DO NOT IMPLEMENT - DIAGNOSIS ONLY)
1. **Add hysteresis** to mode switching (different thresholds for entering/exiting LQR)
2. **Consistent angle wrapping** throughout the system
3. **Energy-based mode switching** instead of pure angle-based
4. **LQR stability checks** before engaging

### Parameter Tuning
1. **Reduce switch angle** for high-inertia systems
2. **Energy controller deadband** adjustment based on system parameters
3. **Force limiting** in LQR mode for large initial errors

## Test Results Summary

| Condition | Mode Switches | Energy Error | Max Force | Result |
|-----------|---------------|--------------|-----------|---------|
| Hanging   | 1/200 steps   | -0.318       | 20.6N     | ✅ SUCCESS |
| Random    | 128/500 steps | 4999         | 500.0N    | ❌ FAILURE |

## Conclusion

The controller architecture is fundamentally sound but lacks robustness to high-energy initial conditions. The mode switching logic creates a positive feedback loop that amplifies disturbances rather than rejecting them. The system needs **hysteresis**, **consistent angle handling**, and **energy-aware mode transitions** to achieve robust performance across all initial conditions. 