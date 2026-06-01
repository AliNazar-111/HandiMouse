"""
Vision Layer: Dynamic coordinates filtering.
Implements the adaptive One Euro Filter algorithm, designed to provide ultra-low jitter 
at low speeds (e.g., precise target selection) and zero lag at high speeds (e.g., rapid sweeps).
"""

import math
import time
from typing import List, Optional, Tuple, Union


class LowPassFilter:
    """
    Standard first-order low-pass filter.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.last_value: Optional[Union[float, List[float]]] = None

    def __call__(self, value: Union[float, List[float]], alpha: Optional[float] = None) -> Union[float, List[float]]:
        if alpha is not None:
            self.alpha = alpha

        if self.last_value is None:
            if isinstance(value, list):
                self.last_value = list(value)
            else:
                self.last_value = value
            return self.last_value

        if isinstance(value, list) and isinstance(self.last_value, list):
            for i in range(len(value)):
                self.last_value[i] = self.alpha * value[i] + (1.0 - self.alpha) * self.last_value[i]
        else:
            self.last_value = self.alpha * value + (1.0 - self.alpha) * self.last_value

        return self.last_value


class OneEuroFilter:
    """
    Adaptive low-pass filter that dynamically alters its cutoff frequency 
    based on the input signal's velocity.

    Math & Parameters:
    - min_cutoff (fc_min): Minimum cutoff frequency (Hz). Higher values decrease lag but increase jitter at rest.
      Default 1.0 Hz provides excellent stability for mouse hovering.
    - beta: Speed coefficient. Higher values adapt quicker to fast motions, reducing lag but increasing high-speed jitter.
      Default 0.007 is a sweet spot for desktop gesture cursor control.
    - d_cutoff (fc_d): Cutoff frequency (Hz) for the derivative (velocity) filter. Typically kept at 1.0 Hz.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # Filters for the signal and its derivative (velocity)
        self.x_filt = LowPassFilter(alpha=1.0)
        self.dx_filt = LowPassFilter(alpha=1.0)

        self.last_time: Optional[float] = None
        self.last_value: Optional[Union[float, List[float]]] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        """Calculate the exponential smoothing factor alpha from cutoff frequency and dt."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(
        self, 
        value: Union[float, List[float], Tuple[float, ...]], 
        timestamp: Optional[float] = None
    ) -> Union[float, List[float]]:
        """
        Filter the incoming value.

        Args:
            value: Scalar or sequence of values to filter (e.g., [x, y] or [x, y, z]).
            timestamp: Optional epoch time in seconds. If None, time.time() is used.

        Returns:
            Union[float, List[float]]: Filtered signal matching the input dimension type.
        """
        current_time = timestamp if timestamp is not None else time.time()

        # Handle tuple inputs by casting to list
        is_sequence = isinstance(value, (list, tuple))
        val_list: Union[float, List[float]] = list(value) if is_sequence else value  # type: ignore

        if self.last_time is None or self.last_value is None:
            self.last_time = current_time
            self.last_value = val_list
            # Initialize inner filters
            self.x_filt(val_list)
            # Derivative is initially 0
            zeros = [0.0] * len(val_list) if isinstance(val_list, list) else 0.0
            self.dx_filt(zeros)
            return val_list

        dt = current_time - self.last_time
        # Prevent division by zero or negative time intervals due to system clock adjustments
        if dt <= 0.0:
            return self.last_value

        # 1. Compute velocity (dx/dt)
        if isinstance(val_list, list) and isinstance(self.last_value, list):
            dx = [(val_list[i] - self.last_value[i]) / dt for i in range(len(val_list))]
        else:
            dx = (val_list - self.last_value) / dt  # type: ignore

        # 2. Filter velocity (low-pass)
        alpha_d = self._alpha(self.d_cutoff, dt)
        filtered_dx = self.dx_filt(dx, alpha_d)

        # 3. Calculate dynamic cutoff frequency based on velocity magnitude
        if isinstance(filtered_dx, list):
            # Euclidean norm for multi-dimensional velocity vector
            speed = math.sqrt(sum(v * v for v in filtered_dx))
        else:
            speed = abs(filtered_dx)

        cutoff = self.min_cutoff + self.beta * speed

        # 4. Filter original signal with adaptive cutoff
        alpha = self._alpha(cutoff, dt)
        filtered_val = self.x_filt(val_list, alpha)

        # Save states for the next iteration
        self.last_time = current_time
        self.last_value = filtered_val

        return filtered_val
