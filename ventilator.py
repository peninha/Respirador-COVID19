import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

"""
# function that returns dy/dt
def ventilator(y, t, alpha, m, x0, u0, uSpeed, uDuration, uStart):
    if t < uStart:
        u = 0
    elif t < uStart+uDuration:
        u = uSpeed*(t-uStart)
    else:
        u = uSpeed*(uDuration)
    x, xDot = y
    dydt = [xDot, alpha/m * (1/(x-u) - 1/(x0-u0))]
    return dydt


# initial condition
n = 400
uSpeed = 0.5
uDuration = 2
uStart = 1
u = np.zeros(n)
t = np.linspace(0, 4, n)
for i in range(len(t)):
    if t[i] < uStart:
        u[i] = 0
    elif t[i] < uStart+uDuration:
        u[i] = uSpeed*(t[i]-uStart)
    else:
        u[i] = uSpeed*(uDuration)


m = 0.7
alpha = 500
u0 = u[0]
x0 = 2
xDot0 = 0
y0 = [x0, xDot0]

h = np.zeros(n)
hdot = np.zeros(n)

# solve ODE
solution = odeint(ventilator, y0, t, args=(alpha, m, x0, u0, uSpeed, uDuration, uStart))


# plot results
plt.plot(t, u, 'r--', label='u(t)')
plt.plot(t, solution[:, 0], 'b', label='h(t)')
plt.plot(t, solution[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
"""

def Vambu(u, VaMax, uMax):
    return -VaMax/uMax * u + VaMax


def uFunction(t, uSpeed, uDuration, uStart, uMax, u0):
    if t < uStart:
        u = u0
    elif t < uStart+uDuration:
        u = uSpeed*(t-uStart)
    else:
        u = uSpeed*(uDuration)
    if u > uMax:
        u = uMax
    return u


n = 400     #passos - Pontos de amostragem

#Êmbolo u do motor de passo
uStart = 1      #s    - Tempo de início do acionamento
uDuration = 2   #s    - Duração do acionamento
u0 = 0          #cm   - Posição inicial
uMax = 10       #cm   - Posição final
uSpeed = 5      #cm/s - Velocidade do êmbolo

PEEP = 5    #cmH20  - Pressão positival final da expiração
PIP = 40    #cmH20  - Pressão de pico na inspiração
UIP = 30    #cmH20  - Pressão de inflexão superior
LIP = 5     #cmH20  - Pressão de inflexão inferior
VUIP = 0.6  #litros - Volulme quando UIP
VLIP = 0.3  #litros - Volume quando LIP
Vt = 0.5    #litros - Tidal Volume = Diferença entre pulmão cheio e vazio
Vm = 0.5    #litros - Volume das mangueiras
VaMax = 3   #litros - Volume Máximo do Ambu

Crs = (VUIP-VLIP)/(UIP-LIP)          #litros/cmH20 - Complascência dinâmica
alpha = PEEP*(VaMax + Vm + PEEP*Crs) #cmH20*litros - coeficiente nRT

Va = np.zeros(n)        #litros - Volume Ambu (t)
Vp = np.zeros(n)        #litros - Volume Pulmão (t)
t = np.linspace(0,4,n)  #s      - Tempo
u = np.zeros(n)         #cm     - Posição êmbolo (t)
P = np.zeros(n)         #cmH20  - Pressão no sistema (t)

for i in range(len(t)):
    u[i] = uFunction(t[i], uSpeed, uDuration, uStart, uMax, u0)
    Va[i] = Vambu(u[i], VaMax, uMax)
    P[i] = (-1*(Va[i]+Vm) + np.sqrt( (Va[i]+Vm)**2 + 4*alpha*Crs ) )/(2*Crs)
    Vp[i] = Crs*P[i]

#PLOT    
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('tempo (s)')
ax1.set_ylabel('Volume (litros)', color=color)
ax1.plot(t, Va, 'b--', label='Vol. Ambu')
ax1.plot(t, Vp, 'b-', label='Vol. Pulmão')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='center left')

color = 'tab:red'
ax2 = ax1.twinx()
ax2.set_ylabel('Pressão (cmH20)', color=color)
ax2.plot(t, P, 'r-', label='Pressão')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='center right')

fig.tight_layout()
#plt.grid()
plt.show()