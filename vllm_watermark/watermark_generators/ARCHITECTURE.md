# Watermark Generators Architecture

This document describes the modular architecture of the watermark generators module.

## Design Principles

The watermark generators module follows these key design principles:

1. **Separation of Concerns**: Each generator algorithm is in its own module
2. **Single Responsibility**: Each file handles one specific watermarking algorithm
3. **Inheritance**: All generators inherit from a common base class
4. **Modularity**: Easy to add new algorithms without modifying existing code
5. **Consistent Interface**: All generators implement the same API

## File Structure

```
vllm_watermark/watermark_generators/
├── __init__.py                 # Main module interface
├── base.py                     # Abstract base class and common functionality
├── gumbel_generator.py         # Gumbel watermarking algorithm
├── openai_generator.py         # OpenAI-style watermarking
├── pf_generator.py             # Prefix-free watermarking
├── maryland_generator.py       # Maryland greenlist watermarking
├── vllm_generator.py          # VLLM integration wrapper
├── README.md                   # User documentation
└── ARCHITECTURE.md            # This file
```

## Module Responsibilities

### `base.py`
- Contains the abstract `WmGenerator` base class
- Defines common interface for all generators
- Implements shared functionality:
  - Token generation pipeline
  - Random number generation and seeding
  - Hash-based context seeding
  - GPU/CPU device handling
  - Text decoding utilities
- Provides legacy `BaseGenerator` for backward compatibility

### `gumbel_generator.py`
- **GumbelGenerator**: Implements Gumbel watermarking
- Uses power-law transformation: `r^(1/p)` where `r` is random and `p` is probability
- Optimized for vLLM integration
- Supports payload encoding through vector shifting

### `openai_generator.py`
- **OpenaiGenerator**: Implements OpenAI's watermarking method
- Similar to Gumbel but with specific implementation details
- Power-law transformation with random vector manipulation

### `pf_generator.py`
- **PFGenerator**: Implements prefix-free watermarking
- Uses exponential transformation: `log(p) - log(r)`
- Better theoretical properties for detection
- Optional watermark disabling with `nowm` flag

### `maryland_generator.py`
- **MarylandGenerator**: Implements Maryland's greenlist approach
- Partitions vocabulary into greenlist and redlist
- Adds bias to greenlist words: `logits[greenlist] += delta`
- Configurable greenlist size (`gamma`) and bias strength (`delta`)

### `vllm_generator.py`
- **VLLMGenerator**: Wrapper for direct VLLM integration
- Encapsulates VLLM engine creation and management
- Provides simplified interface for VLLM-based generation

### `__init__.py`
- Provides clean public interface
- Imports all generator classes
- Defines `__all__` for explicit public API

## Class Hierarchy

```
WmGenerator (abstract base class)
├── GumbelGenerator
├── OpenaiGenerator
├── PFGenerator
├── MarylandGenerator
└── VLLMGenerator

BaseGenerator (legacy, inherits from WmGenerator)
```

## Common Interface

All generators implement the same interface defined by `WmGenerator`:

### Required Methods (abstract)
- `sample_next(logits, ngram_tokens, temperature, top_p)`: Sample next token with watermarking

### Provided Methods (implemented in base)
- `generate(prompts, max_gen_len, temperature, top_p)`: Full text generation pipeline
- `get_seed_rng(input_ids)`: Generate deterministic seed from context
- `hashint(integer_tensor)`: Hash function for seeding

### Common Attributes
- `model`, `tokenizer`: Model and tokenizer instances
- `device`: GPU/CPU device for computations
- `rng`: Random number generator with proper seeding
- `ngram`, `seed`, `payload`: Watermarking parameters

## Watermarking Flow

1. **Context Processing**: Extract n-gram context from generated tokens
2. **Seeding**: Generate deterministic seed from context hash
3. **Random Vector**: Generate vocabulary-sized random vector
4. **Payload Encoding**: Shift random vector by payload amount
5. **Probability Manipulation**: Apply algorithm-specific transformation
6. **Token Selection**: Sample next token from modified distribution

## Extension Points

To add a new watermarking algorithm:

1. Create a new file (e.g., `new_algorithm.py`)
2. Inherit from `WmGenerator`
3. Implement the `sample_next` method
4. Add algorithm-specific parameters to `__init__`
5. Add imports to `__init__.py`
6. Update documentation

Example:
```python
# new_algorithm.py
from .base import WmGenerator

class NewAlgorithmGenerator(WmGenerator):
    def __init__(self, *args, custom_param=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_param = custom_param

    def sample_next(self, logits, ngram_tokens, temperature, top_p):
        # Implement your algorithm here
        pass
```

## Integration with vLLM Core

The generators integrate with the vLLM watermarking system through:

```python
# In core.py
from vllm_watermark.watermark_generators import GumbelGenerator

generator = GumbelGenerator(model, tokenizer, **kwargs)
sampler = CustomSampler(model, generator, debug=debug)
```

## Algorithm Comparison

| Algorithm | Method | Strengths | Use Cases |
|-----------|--------|-----------|-----------|
| Gumbel | Power-law r^(1/p) | Balanced performance | General purpose |
| OpenAI | Power-law variant | Well-studied | Research applications |
| PF | Exponential log(p)-log(r) | Theoretical guarantees | High-security needs |
| Maryland | Greenlist bias | Simple, efficient | Debugging, education |

## Benefits of This Architecture

### Maintainability
- Clear separation between algorithms
- Easy to locate and modify specific implementations
- Shared functionality centralized in base class

### Extensibility
- Simple to add new watermarking algorithms
- Minimal changes required to existing code
- Consistent interface for all algorithms

### Usability
- Clean import structure
- Intuitive API design
- Easy algorithm switching

### Performance
- Shared optimizations in base class
- GPU-optimized implementations
- Efficient random number generation

### Testing
- Each algorithm can be tested independently
- Common functionality tested once in base class
- Easy to create algorithm-specific benchmarks

## Future Considerations

This architecture makes it easy to:
- Add new watermarking algorithms from research
- Implement ensemble watermarking methods
- Create algorithm variants with different parameters
- Optimize specific algorithms for different hardware
- Integrate with new language model frameworks