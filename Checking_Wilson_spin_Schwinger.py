from sympy import *
import numpy as np
import math

# # ---------------------------------------------------------------------------------------------------------------------

# Defining constants and symbols

N = 6

m, a, i, E_0, e, l_0, A, B, C = symbols('m a i E_0 e l_0 A B C')

Z = np.array(symbols(f'sigma_z_1:{2*N+1}')) # symbols(...1:2N+1) returns a tuple of symbols with subscripts from 1 to 2N
plus = np.array(symbols(f'sigma_p_1:{2*N+1}'))
minus = np.array(symbols(f'sigma_m_1:{2*N+1}'))

Z_even = Z[1:len(Z):2]
Z_odd = Z[0:len(Z):2]

plus_even = plus[1:len(plus):2] # middle index is the stop index but exclusive eg if middle index is 5 the last output index is 4
plus_odd = plus[0:len(plus):2]

minus_even = minus[1:len(minus):2]
minus_odd = minus[0:len(minus):2]

# # ---------------------------------------------------------------------------------------------------------------------
#
# # Quantum simulations of lattice gauge theories using Wilson fermions goodnotes page 48 (everything is times 2 to remove floats)
#
# term_1 = -2*i*(m+1/a)*(sum(plus_odd*minus_even)-sum(plus_even*minus_odd))
#
# term_2 = -2*(i/a)*(sum(minus_odd[:-1]*Z_even[:-1]*Z_odd[1:]*plus_even[1:]) - sum(plus_odd[:-1]*Z_even[:-1]*Z_odd[1:]*minus_even[1:]))
#
# term_3 = (a)*(N-1)*E_0**2
#
# term_4 = 0
#
# for n in range(1, N):
#     for k in range(1, n+1):
#         term_4 += 2*a*E_0*e*(Z[(2*k-1)-1] + Z[(2*k)-1]) # the minus 1 in both indices is because indices start from 0
#
# term_5 = 0
#
# for n in range(1, N):
#     for k in range(1, n+1):
#         for q in range(1, n+1):
#             term_5 += (a*e**2)*(Z[(2*k-1)-1] + Z[(2*k)-1])*(Z[(2*q-1)-1] + Z[(2*q)-1])
#
# page_48_full_result = term_1 + term_2 + term_3 + term_4 + term_5
#
# # ---------------------------------------------------------------------------------------------------------------------
#
# # Quantum simulations of lattice gauge theories using Wilson fermions goodnotes page 47 (everything is times 2 to remove floats)
#
# term_1_final = -2*i*(m+1/a)*(sum(plus_odd*minus_even)-sum(plus_even*minus_odd))
#
# term_2_final = -2*(i/a)*(sum(minus_odd[:-1]*Z_even[:-1]*Z_odd[1:]*plus_even[1:]) - sum(plus_odd[:-1]*Z_even[:-1]*Z_odd[1:]*minus_even[1:]))
#
# term_3_final = (a)*(N-1)*E_0**2
#
# term_4_final = 0
#
# for k in range(1, N):
#
#     term_4_final += 2*a*E_0*e*(N-k)*(Z[(2*k-1)-1] + Z[(2*k)-1])
#
# term_5_final = 0
#
# for k in range(1, N):
#     for q in range(k+1, N+1):
#         term_5_final += 2*a*e**2*(N-q)*(Z[(2*k-1) - 1] * Z[(2*q-1) - 1] + Z[(2*k-1) - 1] * Z[(2*q) - 1] + Z[(2*k) - 1] * Z[(2*q-1) - 1] + Z[(2*k) - 1] * Z[(2*q) - 1])
#
# term_6_final = 0
#
# for k in range(1, N):
#         term_6_final += 2*a*e**2*(N-k)*(1 + Z[(2*k-1) - 1] * Z[(2*k) - 1])
#
# page_47_full_final_result = term_1_final + term_2_final + term_3_final + term_4_final + term_5_final + term_6_final
#
# for i in range(len(Z)):
#     page_48_full_result = page_48_full_result.expand().replace(Z[i]**2, 1) # make sigma_z*sigma_z = 1
#
# for i in range(len(Z)):
#     page_47_full_final_result = page_47_full_final_result.expand().replace(Z[i]**2, 1)
#
# # ---------------------------------------------------------------------------------------------------------------------
#
# # Check that the two expressions from page 47 and page 48 agree, this is checking the manipulations of swapping sum orders etc
#
# print('This is page 48 result: ', page_48_full_result, '\n')
# print('This is page 47 result: ', expand(page_47_full_final_result), '\n')
# print('Page 47 == Page 48: ', expand(page_47_full_final_result) == (page_48_full_result), '\n')
#
# # ---------------------------------------------------------------------------------------------------------------------
#
# # Stefan's terms that correspond to red underlined terms in my Quantum simulations of lattice gauge theories using
# # Wilson fermions goodnotes page 50
#
# stefan_1 = 0
#
# for p in range(1, 2*N-1):
#     stefan_1 += 2*l_0*(N-math.ceil(p/2))*Z[p-1]
#
# stefan_2 = 0
#
# for p in range(1, 2*N-1):
#     for p_dash in range(p+1, 2*N-1):
#         stefan_2 += (N-math.ceil(p_dash/2))*Z[p-1]*Z[p_dash-1]
#
# stefan_total = stefan_1 + stefan_2
#
# # ---------------------------------------------------------------------------------------------------------------------
#
# # Red underlined terms from Quantum simulations of lattice gauge theories using Wilson fermions goodnotes page 50
#
# takis_1 = 0
#
# for k in range(1, N):
#     takis_1 += 2*l_0*(N-k)*(Z[(2*k-1)-1] + Z[(2*k)-1])
#
# takis_2 = 0
#
# for k in range(1, N):
#     for q in range(k+1, N):
#         takis_2 += (N-q)*(Z[(2*k-1) - 1] * Z[(2*q-1) - 1] + Z[(2*k-1) - 1] * Z[(2*q) - 1] + Z[(2*k) - 1] * Z[(2*q-1) - 1] + Z[(2*k) - 1] * Z[(2*q) - 1])
#
# takis_3 = 0
#
# for k in range(1, N):
#     takis_3 += (N-k)*Z[(2*k-1)-1]*Z[(2*k)-1]
#
# takis_total = takis_1 + takis_2 + takis_3
#
# for i in range(len(Z)):
#     takis_total = takis_total.expand().replace(Z[i]**2, 1)
#
# for i in range(len(Z)):
#     stefan_total = stefan_total.expand().replace(Z[i]**2, 1)
#
# # ---------------------------------------------------------------------------------------------------------------------
#
# # Check that the red underlined terms in Quantum simulations of lattice gauge theories using Wilson fermions goodnotes
# # page 50 agree with Stefan's corresponding terms
#
# print('Stefan terms: ', stefan_total, '\n')
# print('Takiss terms: ', takis_total, '\n')
# print('Stefan terms == Takis terms: ', expand(takis_total) == expand(stefan_total))
#
# # ---------------------------------------------------------------------------------------------------------------------
#
# # Check that the penalty terms agree, the expression can be found on page 55 of Quantum simulations of lattice gauge
# # theories using Wilson fermions goodnotes
#
# l = 2 # set lambda multiplier to 2 so that the 2 in the denominator cancels and we only have integers
#
# stefan_penalty = N
#
# for p in range(1, 2*N+1):
#     for p_dash in range(p+1, 2*N+1):
#         stefan_penalty += Z[p-1]*Z[p_dash-1]
#
# takis_penalty = N
#
# for q in range(1, N+1):
#     takis_penalty += Z[(2*q-1)-1]*Z[(2*q)-1]
#
# for q in range(1, N):
#     for k in range(q+1, N+1):
#         takis_penalty += Z[(2*k-1)-1]*Z[(2*q-1)-1] + Z[(2*k-1)-1]*Z[(2*q)-1] + Z[(2*k)-1]*Z[(2*q-1)-1] + Z[(2*k)-1]*Z[(2*q)-1]
#
# for i in range(len(Z)):
#     takis_penalty = takis_penalty.expand().replace(Z[i]**2, 1)
#
# for i in range(len(Z)):
#     stefan_penalty = stefan_penalty.expand().replace(Z[i]**2, 1)
#
# print(stefan_penalty)
# print(takis_penalty)
# print(takis_penalty == stefan_penalty)
#
# # ---------------------------------------------------------------------------------------------------------------------
#
# This is checking the pages 5 and 6 of Wilson MPO in goodnotes
#
# mpo = {i:None for i in range(1, 2*N+1)}
#
# for key in mpo.keys():
#
#     if key == 1:
#         mpo[key] = np.array([[0, symbols(f'sigma_z_{key}'), 1]])
#     elif key == 2*N:
#         mpo[key] = np.array([[1, 0.5*(N-key/2+symbols('lambda'))*symbols(f'sigma_z_{key}'), 0]]).T
#     else:
#
#         if key % 2 == 0:
#             if key == 2*N:
#                 mpo[key] = np.array([[1, 0, 0], [0.5*(N-key/2 + symbols('lambda'))*symbols(f'sigma_z_{key}'), 1, 0], [0, 0, 1]])
#             else:
#                 mpo[key] = np.array([[1, 0, 0], [0.5 * (N - key / 2 + symbols('lambda')) * symbols(f'sigma_z_{key}'), 1, 0], [0, symbols(f'sigma_z_{key}'), 1]])
#         else:
#             mpo[key] = np.array([[1, 0, 0], [0.5 * (N - key / 2 - 1/2 + symbols('lambda')) * symbols(f'sigma_z_{key}'), 1, 0], [0, symbols(f'sigma_z_{key}'), 1]])
#
#
# result = mpo[1]
# for i in range(2, 2*N+1):
#     result = np.matmul(result, mpo[i])
#
# double_terms = 0
#
# for k in range(1, N):
#     for q in range(k+1, N+1):
#         double_terms += 0.5*(N-q+symbols('lambda'))*(Z[(2*k-1)-1]*Z[(2*q-1)-1] + Z[(2*k-1)-1]*Z[(2*q)-1] + Z[(2*k)-1]*Z[(2*q-1)-1] + Z[(2*k)-1]*Z[(2*q)-1])
#
# for k in range(1, N+1):
#     double_terms += 0.5*(N-k + symbols('lambda'))*(Z[(2*k-1)-1]*Z[(2*k)-1])
#
#
# for i in range(len(Z)):
#     double_terms = double_terms.expand().replace(Z[i]**2, 1)
#
# result = result[0][0]
#
# for i in range(len(Z)):
#     result = result.expand().replace(Z[i]**2, 1)
#
# print(result)
# print(double_terms)
# print(result == double_terms)
#
# # ---------------------------------------------------------------------------------------------------------------------

# Checking W dash's final expression with the MPO

# mpo = {i:None for i in range(1, 2*N+1)}
#
# for key in mpo.keys():
#
#     if key == 1:
#         mpo[key] = np.array([[C/(2*N)+l_0*(N-key/2-1/2)*symbols(f'sigma_z_{key}'), A*symbols(f'sigma_p_{key}'), -A*symbols(f'sigma_m_{key}'), B*symbols(f'sigma_m_{key}'), -B*symbols(f'sigma_p_{key}'), symbols(f'sigma_z_{key}'), 1]])
#     elif key == 2*N:
#         mpo[key] = np.array([[1,symbols(f'sigma_m_{key}'),symbols(f'sigma_p_{key}'),0,0,0.5*(N-key/2+symbols('lambda'))*symbols(f'sigma_z_{key}'),C/(2*N)+l_0*(N-key/2)*symbols(f'sigma_z_{key}')]]).T
#     else:
#
#         if key % 2 == 0:
#             if key == 2*N:
#                 mpo[key] = np.array([[1, 0, 0, 0, 0, 0, 0],[symbols(f'sigma_m_{key}'), 0, 0, 0, 0, 0, 0],[symbols(f'sigma_p_{key}'), 0, 0, 0, 0, 0, 0],[0, 0, symbols(f'sigma_z_{key}'), 0, 0, 0, 0],[0, symbols(f'sigma_z_{key}'), 0, 0, 0, 0, 0],[0.5*(N-key/2+symbols('lambda'))*symbols(f'sigma_z_{key}'), 0, 0, 0, 0, 1, 0],[C/(2*N)+l_0*(N-key/2)*symbols(f'sigma_z_{key}'), 0, 0, 0, 0, 0, 1]])
#             else:
#                 mpo[key] = np.array([[1, 0, 0, 0, 0, 0, 0],[symbols(f'sigma_m_{key}'), 0, 0, 0, 0, 0, 0],[symbols(f'sigma_p_{key}'), 0, 0, 0, 0, 0, 0],[0, 0, symbols(f'sigma_z_{key}'), 0, 0, 0, 0],[0, symbols(f'sigma_z_{key}'), 0, 0, 0, 0, 0],[0.5*(N-key/2+symbols('lambda'))*symbols(f'sigma_z_{key}'), 0, 0, 0, 0, 1, 0],[C/(2*N)+l_0*(N-key/2)*symbols(f'sigma_z_{key}'), 0, 0, 0, 0, symbols(f'sigma_z_{key}'), 1]])
#         else:
#             mpo[key] = np.array([[1, 0, 0, 0, 0, 0, 0],[0, symbols(f'sigma_z_{key}'), 0, 0, 0, 0, 0],[0, 0, symbols(f'sigma_z_{key}'), 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0.5*(N-key/2-1/2+symbols('lambda'))*symbols(f'sigma_z_{key}'), 0, 0, 0, 0, 1, 0],[C/(2*N)+l_0*(N-key/2-1/2)*symbols(f'sigma_z_{key}'), A*symbols(f'sigma_p_{key}'), -A*symbols(f'sigma_m_{key}'), B*symbols(f'sigma_m_{key}'), -B*symbols(f'sigma_p_{key}'), symbols(f'sigma_z_{key}'), 1]])
#
# result = mpo[1]
# for i in range(2, 2*N+1):
#     result = np.matmul(result, mpo[i])
#
# result = result[0][0]
# for i in range(len(Z)):
#     result = result.expand().replace(Z[i]**2, 1)
#
# for a in preorder_traversal(result):
#     if isinstance(a, Integer):
#         result = result.subs(a, float(a))
#
# result = result.replace(1.0, symbols('R'))
# result = result.replace(symbols('R'), 1)
#
# w_dash = 0
#
# for n in range(1, N+1):
#     w_dash += A*(plus[(2*n-1)-1]*minus[(2*n)-1] - minus[(2*n-1)-1]*plus[(2*n)-1])
#
# for n in range(1, N):
#     w_dash += B*(minus[(2*n-1)-1]*Z[(2*n)-1]*Z[(2*n+1)-1]*plus[(2*n+2)-1] - plus[(2*n-1)-1]*Z[(2*n)-1]*Z[(2*n+1)-1]*minus[(2*n+2)-1])
#
# for k in range(1, N):
#     w_dash += l_0*(N-k)*(Z[(2*k-1)-1] + Z[(2*k)-1])
#
# for k in range(1, N):
#     for q in range(k+1, N+1):
#         w_dash += 0.5*(N-q+symbols('lambda'))*(Z[(2*k-1)-1]*Z[(2*q-1)-1] + Z[(2*k-1)-1]*Z[(2*q)-1] + Z[(2*k)-1]*Z[(2*q-1)-1] + Z[(2*k)-1]*Z[(2*q)-1])
#
# for k in range(1, N+1):
#     w_dash += 0.5*(N-k+symbols('lambda'))*(Z[(2*k-1)-1]*Z[(2*k)-1])
#
# w_dash = w_dash + C
#
#
# for i in range(len(Z)):
#     w_dash = w_dash.expand().replace(Z[i]**2, 1)
#
#
# for a in preorder_traversal(w_dash):
#     if isinstance(a, Integer):
#         w_dash = w_dash.subs(a, float(a))
#
#
# w_dash = w_dash.replace(1.0, symbols('Y'))
# w_dash = w_dash.replace(symbols('Y'), 1)
#
# print(result)
# print(w_dash)
# print(result == w_dash)
# print(len(str(result).split('+')))
# print(len(str(w_dash).split('+')))

# ---------------------------------------------------------------------------------------------------------------------

# Checking that W' and MPO all agree for both Stefan and mine for all possible combination checks

mpo = {i:None for i in range(1, 2*N+1)}

for key in mpo.keys():

    if key == 1:
        mpo[key] = np.array([[C/(2*N)+l_0*(N-key/2-1/2)*symbols(f'sigma_z_{key}'), A*symbols(f'sigma_p_{key}'), -A*symbols(f'sigma_m_{key}'), B*symbols(f'sigma_m_{key}'), -B*symbols(f'sigma_p_{key}'), symbols(f'sigma_z_{key}'), 1]])
    elif key == 2*N:
        mpo[key] = np.array([[1,symbols(f'sigma_m_{key}'),symbols(f'sigma_p_{key}'),0,0,0.5*(N-key/2+symbols('lambda'))*symbols(f'sigma_z_{key}'),C/(2*N)+l_0*(N-key/2)*symbols(f'sigma_z_{key}')]]).T
    else:

        if key % 2 == 0:
            mpo[key] = np.array([[1, 0, 0, 0, 0, 0, 0],[symbols(f'sigma_m_{key}'), 0, 0, 0, 0, 0, 0],[symbols(f'sigma_p_{key}'), 0, 0, 0, 0, 0, 0],[0, 0, symbols(f'sigma_z_{key}'), 0, 0, 0, 0],[0, symbols(f'sigma_z_{key}'), 0, 0, 0, 0, 0],[0.5*(N-key/2+symbols('lambda'))*symbols(f'sigma_z_{key}'), 0, 0, 0, 0, 1, 0],[C/(2*N)+l_0*(N-key/2)*symbols(f'sigma_z_{key}'), 0, 0, 0, 0, symbols(f'sigma_z_{key}'), 1]])
        else:
            mpo[key] = np.array([[1, 0, 0, 0, 0, 0, 0],[0, symbols(f'sigma_z_{key}'), 0, 0, 0, 0, 0],[0, 0, symbols(f'sigma_z_{key}'), 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0.5*(N-key/2-1/2+symbols('lambda'))*symbols(f'sigma_z_{key}'), 0, 0, 0, 0, 1, 0],[C/(2*N)+l_0*(N-key/2-1/2)*symbols(f'sigma_z_{key}'), A*symbols(f'sigma_p_{key}'), -A*symbols(f'sigma_m_{key}'), B*symbols(f'sigma_m_{key}'), -B*symbols(f'sigma_p_{key}'), symbols(f'sigma_z_{key}'), 1]])

result = mpo[1]
for i in range(2, 2*N+1):
    result = np.matmul(result, mpo[i])

result = result[0][0]
for i in range(len(Z)):
    result = result.expand().replace(Z[i]**2, 1)

for a in preorder_traversal(result):
    if isinstance(a, Integer):
        result = result.subs(a, float(a))

result = result.replace(1.0, symbols('R'))
result = result.replace(symbols('R'), 1)

w_dash_stefan = 0

for n in range(1, N+1):
    w_dash_stefan += A*(plus[(2*n-1)-1]*minus[(2*n)-1] - minus[(2*n-1)-1]*plus[(2*n)-1])

for n in range(1, N):
    w_dash_stefan += B*(minus[(2*n-1)-1]*Z[(2*n)-1]*Z[(2*n+1)-1]*plus[(2*n+2)-1] - plus[(2*n-1)-1]*Z[(2*n)-1]*Z[(2*n+1)-1]*minus[(2*n+2)-1])

for p in range(1, 2*N-1):
    w_dash_stefan += l_0*(N-math.ceil(p/2))*Z[p-1]

for p in range(1, 2*N-1):
    for p_dash in range(p+1, 2*N-1):
        w_dash_stefan += 0.5*(N-math.ceil(p_dash/2))*Z[p-1]*Z[p_dash-1]

for p in range(1, 2*N+1):
    for p_dash in range(p+1, 2*N+1):
        w_dash_stefan += 0.5*(symbols('lambda'))*Z[p-1]*Z[p_dash-1]

w_dash_stefan += C

w_dash = 0

for n in range(1, N+1):
    w_dash += A*(plus[(2*n-1)-1]*minus[(2*n)-1] - minus[(2*n-1)-1]*plus[(2*n)-1])

for n in range(1, N):
    w_dash += B*(minus[(2*n-1)-1]*Z[(2*n)-1]*Z[(2*n+1)-1]*plus[(2*n+2)-1] - plus[(2*n-1)-1]*Z[(2*n)-1]*Z[(2*n+1)-1]*minus[(2*n+2)-1])

for k in range(1, N):
    w_dash += l_0*(N-k)*(Z[(2*k-1)-1] + Z[(2*k)-1])

for k in range(1, N):
    for q in range(k+1, N+1):
        w_dash += 0.5*(N-q+symbols('lambda'))*(Z[(2*k-1)-1]*Z[(2*q-1)-1] + Z[(2*k-1)-1]*Z[(2*q)-1] + Z[(2*k)-1]*Z[(2*q-1)-1] + Z[(2*k)-1]*Z[(2*q)-1])

for k in range(1, N+1):
    w_dash += 0.5*(N-k+symbols('lambda'))*(Z[(2*k-1)-1]*Z[(2*k)-1])

w_dash = w_dash + C

for i in range(len(Z)):
    w_dash_stefan = w_dash_stefan.expand().replace(Z[i]**2, 1)

for i in range(len(Z)):
    w_dash = w_dash.expand().replace(Z[i]**2, 1)

for a in preorder_traversal(w_dash):
    if isinstance(a, Integer):
        w_dash = w_dash.subs(a, float(a))

for a in preorder_traversal(w_dash_stefan):
    if isinstance(a, Integer):
        w_dash_stefan = w_dash_stefan.subs(a, float(a))

w_dash = w_dash.replace(1.0, symbols('R'))
w_dash = w_dash.replace(symbols('R'), 1)

w_dash_stefan = w_dash_stefan.replace(1.0, symbols('R'))
w_dash_stefan = w_dash_stefan.replace(symbols('R'), 1)

mpo_stefan = {i:None for i in range(1, 2*N+1)}

for key in mpo_stefan.keys():

    if key == 1:
        mpo_stefan[key] = np.array([[1,symbols(f'sigma_p_{key}'),0,symbols(f'sigma_m_{key}'),0,symbols(f'sigma_z_{key}'),C/(2*N) + l_0*(N-math.ceil(key/2))*symbols(f'sigma_z_{key}')]])
    elif key == 2*N:
        mpo_stefan[key] = np.array([[C/(2*N) + l_0*symbols(f'sigma_z_{key}')*(N-math.ceil(key/2)),A*symbols(f'sigma_m_{key}'),-B*symbols(f'sigma_m_{key}'),-A*symbols(f'sigma_p_{key}'),B*symbols(f'sigma_p_{key}'),0.5*(symbols('lambda'))*symbols(f'sigma_z_{key}'),1]]).T
    else:

        if key % 2 == 0:
            mpo_stefan[key] = np.array([[1,0,0,0,0,symbols(f'sigma_z_{key}'),C/(2*N) + l_0*symbols(f'sigma_z_{key}')*(N-math.ceil(key/2))],[0,symbols(f'sigma_z_{key}'),0,0,0,0,A*symbols(f'sigma_m_{key}')],[0,0,0,0,0,0,-B*symbols(f'sigma_m_{key}')],[0,0,0,symbols(f'sigma_z_{key}'),0,0,-A*symbols(f'sigma_p_{key}')],[0,0,0,0,0,0,B*symbols(f'sigma_p_{key}')],[0,0,0,0,0,1,0.5*(N-math.ceil(key/2)+symbols('lambda'))*symbols(f'sigma_z_{key}')],[0,0,0,0,0,0,1]])
        else:
            mpo_stefan[key] = np.array([[1,symbols(f'sigma_p_{key}'),0,symbols(f'sigma_m_{key}'),0,symbols(f'sigma_z_{key}'),C/(2*N) + l_0*(N-math.ceil(key/2))*symbols(f'sigma_z_{key}')],[0,0,symbols(f'sigma_z_{key}'),0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,symbols(f'sigma_z_{key}'),0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,1,(N-math.ceil(key/2)+symbols('lambda'))*0.5*symbols(f'sigma_z_{key}')],[0,0,0,0,0,0,1]])

result_stefan = mpo_stefan[1]
for i in range(2, 2*N+1):
    result_stefan = np.matmul(result_stefan, mpo_stefan[i])

result_stefan = result_stefan[0][0]
for i in range(len(Z)):
    result_stefan = result_stefan.expand().replace(Z[i]**2, 1)

for a in preorder_traversal(result_stefan):
    if isinstance(a, Integer):
        result_stefan = result_stefan.subs(a, float(a))

result_stefan = result_stefan.replace(1.0, symbols('R'))
result_stefan = result_stefan.replace(symbols('R'), 1)

print(w_dash)
print(w_dash_stefan)
print(result)
print(result_stefan)
print(len(str(w_dash).split('+')))
print(len(str(w_dash_stefan).split('+')))
print(len(str(result).split('+')))
print(w_dash == w_dash_stefan)
print(result == w_dash_stefan)
print(result == w_dash)
print(result_stefan == w_dash_stefan)
print(result_stefan == w_dash)

# # -------------------------------------------------------------------------------------------------------------------
