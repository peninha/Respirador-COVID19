# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:39:35 2020

@author: Pena
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def pend(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt


b = 0.25
c = 5.0
y0 = [np.pi - 0.1, 0.0]

t = np.linspace(0, 10, 101)

sol = odeint(pend, y0, t, args=(b, c))

plt.plot(t, sol[:, 0], 'b', label='h(t)')
plt.plot(t, sol[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()