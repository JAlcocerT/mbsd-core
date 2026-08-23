"""
Parameterized surfaces for CAM constraints.

A surface is defined by a curve with parameter s (arc length, angle, etc.)
Each surface provides:
  - position(s): point on the curve at parameter s
  - tangent(s): tangent vector (first derivative)
  - curvature(s): curvature (second derivative, optional)
"""

import numpy as np
from abc import ABC, abstractmethod


class Surface(ABC):
    """Abstract base class for parameterized surfaces."""

    @abstractmethod
    def position(self, s):
        """Return point on surface at parameter s: [x, y]."""
        pass

    @abstractmethod
    def tangent(self, s):
        """Return tangent vector (first derivative) at s: [dx/ds, dy/ds]."""
        pass

    def curvature(self, s):
        """Return second derivative (curvature info) at s: [d²x/ds², d²y/ds²]."""
        # Numerical differentiation if not overridden
        h = 1e-6
        t_plus = self.tangent(s + h)
        t_minus = self.tangent(s - h)
        return (t_plus - t_minus) / (2 * h)


class CircleSurface(Surface):
    """Circle with radius R, parameterized by angle."""

    def __init__(self, radius):
        self.R = radius

    def position(self, s):
        """s is angle in radians."""
        return np.array([self.R * np.cos(s), self.R * np.sin(s)])

    def tangent(self, s):
        """Tangent vector (perpendicular to radius)."""
        return np.array([-self.R * np.sin(s), self.R * np.cos(s)])

    def curvature(self, s):
        """Second derivative."""
        return np.array([-self.R * np.cos(s), -self.R * np.sin(s)])


class LineSurface(Surface):
    """Straight line, parameterized by distance along line."""

    def __init__(self, direction):
        """direction: unit vector along the line."""
        self.d = np.array(direction) / np.linalg.norm(direction)

    def position(self, s):
        """s is distance along line from origin."""
        return s * self.d

    def tangent(self, s):
        """Tangent is constant along the line."""
        return self.d.copy()

    def curvature(self, s):
        """Second derivative is zero (line is straight)."""
        return np.array([0.0, 0.0])


class PolynomialSurface(Surface):
    """Polynomial curve: x(s) = p_x(s), y(s) = p_y(s).

    Parameterized by arc length (approximately).
    """

    def __init__(self, poly_x, poly_y):
        """
        poly_x, poly_y: numpy polynomials (use numpy.poly1d)

        Example:
          px = np.poly1d([1, 0, 0])  # x = s^2
          py = np.poly1d([0, 1])     # y = s
          surf = PolynomialSurface(px, py)
        """
        self.poly_x = poly_x
        self.poly_y = poly_y

    def position(self, s):
        """Evaluate polynomials at s."""
        return np.array([float(self.poly_x(s)), float(self.poly_y(s))])

    def tangent(self, s):
        """Derivative of polynomials."""
        dpx = self.poly_x.deriv()
        dpy = self.poly_y.deriv()
        return np.array([float(dpx(s)), float(dpy(s))])

    def curvature(self, s):
        """Second derivative."""
        d2px = self.poly_x.deriv(2)
        d2py = self.poly_y.deriv(2)
        return np.array([float(d2px(s)), float(d2py(s))])


class SplineSurface(Surface):
    """Smooth spline through control points, parameterized by arc length.

    Uses cubic Hermite spline interpolation between points.
    """

    def __init__(self, points, closed=False):
        """
        points: list of [x, y] points to interpolate
        closed: if True, curve wraps around (for closed CAMs)
        """
        self.points = np.array(points)
        self.closed = closed
        self.n = len(points)

        if closed:
            # Append first point at end for wrapping
            self.points = np.vstack([self.points, self.points[0:1]])

    def _get_segment(self, s):
        """Get segment index and local parameter for parameter s."""
        # Simple arc length parameterization: s in [0, n-1)
        seg = int(s) % self.n
        t = s - int(s)
        return seg, t

    def position(self, s):
        """Cubic Hermite interpolation."""
        seg, t = self._get_segment(s)
        p0 = self.points[seg]
        p1 = self.points[seg + 1]

        # Use simple linear interpolation (can upgrade to Hermite)
        return (1 - t) * p0 + t * p1

    def tangent(self, s):
        """Numerical differentiation of position."""
        h = 1e-6
        return (self.position(s + h) - self.position(s - h)) / (2 * h)

    def curvature(self, s):
        """Numerical differentiation of tangent."""
        h = 1e-6
        return (self.tangent(s + h) - self.tangent(s - h)) / (2 * h)
