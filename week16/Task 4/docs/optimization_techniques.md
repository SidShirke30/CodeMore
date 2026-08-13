# Model Optimization Techniques

## Pruning
The optimized Random Forest uses fewer trees (80 instead of 200) and a maximum tree depth of 8. This reduces structural complexity and can lower inference cost.

## Quantization
Quantization represents model parameters using lower precision, such as INT8 instead of FP32. It can reduce memory usage and improve inference speed on compatible hardware. It is documented here as an additional production technique.

## Knowledge Distillation
A large teacher model can generate targets for a smaller student model. The student learns to approximate the teacher while requiring fewer resources.

## Evaluation
Every optimization should be evaluated using:
- accuracy or another task-specific quality metric
- inference latency
- model size
- CPU and memory utilization

The benchmark measures accuracy, latency, and serialized model size.
