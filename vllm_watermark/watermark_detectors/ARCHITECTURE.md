# Watermark Detectors Architecture

This document describes the modular architecture of the watermark detectors module.

## Design Principles

The watermark detectors module follows these key design principles:

1. **Separation of Concerns**: Each detector type is in its own module
2. **Single Responsibility**: Each file handles one specific type of detector
3. **Inheritance**: All detectors inherit from a common base class
4. **Modularity**: Easy to add new detector types without modifying existing code

## File Structure

```
vllm_watermark/watermark_detectors/
├── __init__.py                 # Main module interface
├── base.py                     # Abstract base class
├── maryland_detectors.py       # Maryland-style detectors
├── openai_detectors.py        # OpenAI-style detectors
├── pf_detector.py             # Prefix-free detector
├── README.md                   # User documentation
└── ARCHITECTURE.md            # This file
```

## Module Responsibilities

### `base.py`
- Contains the abstract `WmDetector` base class
- Defines common interface for all detectors
- Implements shared functionality:
  - Token scoring infrastructure
  - Text processing pipeline
  - P-value calculation framework
  - Detection result formatting

### `maryland_detectors.py`
- **MarylandDetector**: Implements binomial distribution-based detection
- **MarylandDetectorZ**: Implements z-score approximation variant
- Both use greenlist-based scoring approach

### `openai_detectors.py`
- **OpenaiDetector**: Implements gamma distribution-based detection
- **OpenaiDetectorZ**: Implements z-score approximation variant
- Both use exponential scoring based on random values

### `pf_detector.py`
- **PFDetector**: Implements prefix-free watermark detection
- Uses gamma distribution with special handling for zero values

### `__init__.py`
- Provides clean public interface
- Imports all detector classes
- Defines `__all__` for explicit public API

## Class Hierarchy

```
WmDetector (abstract base class)
├── MarylandDetector
├── MarylandDetectorZ
├── OpenaiDetector
├── OpenaiDetectorZ
└── PFDetector
```

## Common Interface

All detectors implement the same interface defined by `WmDetector`:

### Required Methods (abstract)
- `score_tok(ngram_tokens, token_id)`: Score a single token
- `get_pvalue(score, ntoks, eps)`: Calculate statistical p-value

### Provided Methods (implemented in base)
- `detect(text)`: Main detection interface
- `get_scores_by_t(texts, ...)`: Batch scoring
- `get_pvalues(scores, ...)`: Batch p-value calculation
- `aggregate_scores(scores, ...)`: Score aggregation

## Extension Points

To add a new detector type:

1. Create a new file (e.g., `new_detector.py`)
2. Inherit from `WmDetector`
3. Implement required abstract methods
4. Add imports to `__init__.py`
5. Update documentation

Example:
```python
# new_detector.py
from .base import WmDetector

class NewDetector(WmDetector):
    def score_tok(self, ngram_tokens, token_id):
        # Implement scoring logic
        pass

    def get_pvalue(self, score, ntoks, eps=1e-200):
        # Implement p-value calculation
        pass
```

## Benefits of This Architecture

### Maintainability
- Changes to one detector don't affect others
- Clear separation of concerns
- Easy to locate and fix issues

### Extensibility
- Simple to add new detector types
- Minimal changes required to existing code
- Consistent interface for all detectors

### Usability
- Clean import structure
- Consistent API across all detectors
- Easy to understand and use

### Testing
- Each detector can be tested independently
- Common functionality tested once in base class
- Easier to write focused unit tests

## Import Patterns

### Import specific detectors
```python
from vllm_watermark.watermark_detectors import MarylandDetectorZ
```

### Import multiple detectors
```python
from vllm_watermark.watermark_detectors import (
    MarylandDetectorZ,
    OpenaiDetectorZ,
)
```

### Import base class for extension
```python
from vllm_watermark.watermark_detectors.base import WmDetector
```

## Future Considerations

This architecture makes it easy to:
- Add new watermarking algorithms and their corresponding detectors
- Implement ensemble detection methods
- Add specialized detectors for specific use cases
- Create detector variants with different statistical approaches