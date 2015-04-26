__author__ = 'James L Conaway II'

import numpy as np
import sys

PRECISION = 1e-8
MAX_ITER = 1e5

if len(sys.argv) != 8:
    print('Incorrect number of arguments. Please enter data as follows:')
    print('>python gauss_solver.py r1i r1j r1k r2i r2j r2k t')
    exit(1)

pos1 = np.array([float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3])])
pos2 = np.array([float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6])])
t = float(sys.argv[7])

r1 = np.linalg.norm(pos1)
r2 = np.linalg.norm(pos2)

d_nu = np.arccos(np.dot(pos1, pos2)/(r1*r2))

y = 1.0
x = 0.0
y_prev = 0.0
iter = 0

s = (r1+r2)/(4*np.sqrt(r1*r2)*np.cos(d_nu/2.0))-0.5
w = t**2/(2*np.sqrt(r1*r2)*np.cos(d_nu/2.0))**3

while (np.absolute(y - y_prev) > PRECISION) and (iter < MAX_ITER):
    y_prev = y
    x = w/y**2 - s
    X = (4/3)*(1+6*x/5+48*x**2/35+480*x**3/315+5760*x**4/3465)
    y = 1+X*(s+x)
    iter += 1

if iter >= MAX_ITER:
    print('Maximum iterations reached. Solution cannot be determined.')
    exit(2)

U = 1-2*(w/y**2-s)
p = (r1*r2*(1-np.cos(d_nu)))/(r1+r2-2*np.sqrt(r1*r2)*np.cos(d_nu/2)*U)

f = 1-r2*(1-np.cos(d_nu))/p
g = r1*r2*np.sin(d_nu)/np.sqrt(p)
f_dot = np.sqrt(1/p)*np.tan(d_nu/2)*((1-np.cos(d_nu))/p-1/r1-1/r2)
g_dot = 1-r1*(1-np.cos(d_nu))/p

vel1 = (pos2-f*pos1)/g
vel2 = f_dot*pos1+g_dot*vel1

h_vec = np.cross(pos1,vel1)
e_vec = np.cross(vel1,h_vec)-pos1/r1

print('v1 = {0}'.format(vel1))
print('v2 = {0}'.format(vel2))
print('h  = {0}'.format(h_vec))
print('e  = {0}'.format(e_vec))
print('p  = {0:.8f}'.format(p))