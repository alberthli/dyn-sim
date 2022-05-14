from abc import ABC, ABCMeta, abstractmethod

import numpy as np

class System(ABC):
	"""Abstract class for dynamical systems."""

	def __init__(self, n: int, m: int) -> None:
		"""Initialize a system.

		Parameters
		----------
		n : int
			Dimension of state.
		m : int
			Dimension of control input.
		"""
		super(System, self).__init__()

		assert n > 0
		assert m >= 0

		self._n = n
		self._m = m

	@abstractmethod
	def dyn(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
		"""Dynamics of the system.

		Parameters
		----------
		t : float
			Current time.
		x : np.ndarray, shape=(n,)
			Current state.
		u : np.ndarray, shape=(m,)
			Current control input.

		Returns
		-------
		dx : np.ndarray, shape=(n,)
			Time derivative of current state.
		"""
		pass

class CtrlAffineSystem(System, metaclass=ABCMeta):
	"""Abstract class for control affine system."""

	def __init__(self, n: int, m: int) -> None:
		"""Initialize a control affine system.

		The dynamics should have the form:
		dx = f(t, x) + g(t, x)u

		Parameters
		----------
		n : int
			Dimension of state.
		m : int
			Dimension of control input.
		"""
		super(CtrlAffineSystem, self).__init__()

	@abstractmethod
	def f(self, t: float, x: np.ndarray) -> np.ndarray:
		"""Drift of the system.

		Parameters
		----------
		t : float
			Current time.
		x : np.ndarray, shape=(n,)
			Current state.

		Returns
		-------
		f_val : np.ndarray, shape=(n,)
			Value of the drift of the system.
		"""
		pass

	@abstractmethod
	def g(self, t: float, x: np.ndarray) -> np.ndarray:
		"""Actuation matrix.

		Parameters
		----------
		t : float
			Current time.
		x : np.ndarray, shape=(n,)
			Current state.

		Returns
		-------
		g_val : np.ndarray, shape=(n, m)
			Value of the actuation matrix of the system.
		"""
		pass

	def dyn(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
		"""See parent docstring."""
		assert x.shape == (self._n,)
		assert u.shape == (self._m,)
		return f(t, x) + g(t, x) * u
