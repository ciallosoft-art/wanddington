"""Entry point for `python -m waddington_select`.

Runs the sequential oracle evaluation (the paper experiment runner).
For the cross-experiment memory builder, use `python -m waddington_select.memory_builder`.
"""

from .run_sequential import main

if __name__ == "__main__":
    main()
