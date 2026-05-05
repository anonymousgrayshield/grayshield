# Table 2: Attacker Variant Robustness

| Defense | NAIVE | REPEAT3 | REPEAT5 | INTERLEAVE | RS |
|---------|------|------|------|------|------|
| **FineTune** | 45.2±1.4 | 45.5±1.0 | 46.3±0.9 | 45.3±1.3 | 45.2±1.1 |
| **GaussianNoise** | 37.5±6.1 | 38.4±5.6 | 40.3±4.9 | 37.7±6.0 | 37.5±6.0 |
| **GrayShield** | 49.5±0.7 | 50.4±0.6 | 50.2±0.2 | 50.0±0.2 | 49.6±0.8 |
| **PTQ** | 50.0±0.5 | 49.5±0.5 | 50.1±0.5 | 50.0±0.5 | 49.5±0.4 |
| **PatternMask** | 50.0±3.2 | 50.0±3.3 | 50.0±3.3 | 50.0±3.2 | 50.0±3.2 |
| **RandomFlip** | 2.7±3.7 | 0.6±1.0 | 0.2±0.3 | 2.7±3.7 | 2.5±3.8 |
| **SWP** | 10.0±2.1 | 9.5±0.9 | 8.6±0.9 | 10.1±2.1 | 5.5±2.7 |

**Interpretation:**
- Values show mean ± std deviation of recovery reduction (%)
- **Higher values** indicate better defense (harder payload recovery)
- **Low variance** across variants indicates robustness to adaptive attacks