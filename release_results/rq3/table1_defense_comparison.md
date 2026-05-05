# Table 1: Defense Strategy Comparison (RQ3 Summary)

| Defense | Avg Recovery Reduction | Std Dev | Avg Acc Drop | Avg Defense Time | Consistency¹ |
|---------|------------------------|---------|--------------|------------------|--------------|
| **PatternMask** | 50.00% | 3.25% | 0.134% | 44 ms | 94/100 |
| **GrayShield** | 49.96% | 0.66% | -0.098% | 54 ms | 99/100 |
| **PTQ** | 49.83% | 0.54% | 0.166% | 882 ms | 99/100 |
| **FineTune** | 45.51% | 1.23% | 0.449% | 13221 ms | 97/100 |
| **GaussianNoise** | 38.25% | 5.85% | -0.037% | 620 ms | 85/100 |
| **SWP** | 8.76% | 2.56% | 0.068% | 1483 ms | 71/100 |
| **RandomFlip** | 1.73% | 3.16% | 0.018% | 2590 ms | 0/100 |

¹ Consistency Score: Measure of stability across models (100 = perfectly stable)

**Key Findings:**
- **GrayShield** remains the most stable near-chance-level sanitizer across attacker variants
- **PTQ** and **SWP** now provide paper-aligned comparison points without changing the core pipeline
- **RandomFlip** remains a weak defense and serves mainly as a stochastic baseline