from p1afempy.data_structures import CoordinatesType
import numpy as np


def f(r: CoordinatesType) -> float:
    """returns -((d/dx)^2 + (d/dy)^2)analytical(x,y)"""
    return 13.*np.pi**2*np.sin(2.*np.pi * r[:, 0])*np.sin(3.*np.pi * r[:, 1])


def grad_u(r: CoordinatesType) -> np.ndarray:
    xs = r[:, 0]
    ys = r[:, 1]
    tmp_x = np.cos(2. * np.pi * xs) * np.sin(3. * np.pi * ys)
    tmp_y = np.sin(2. * np.pi * xs) * np.cos(3. * np.pi * ys)
    return np.pi * np.column_stack([2.*tmp_x, 3.*tmp_y])