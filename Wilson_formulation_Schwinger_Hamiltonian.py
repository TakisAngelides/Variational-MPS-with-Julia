import numpy as np

# s_list = []
# N_max = 100
# N_list = np.arange(2, 100, 2)
# for N in N_list:
#     s = 0
#     for n in range(1, N):
#         for k in range(1, n+1):
#             for m in range(1, k):
#                 s += (-1)**(m+k)
#     s_list.append(s)
#
# theory = -N_list**2/4 + N_list/2 - 1/8 - 1/8 * (-1)**(N_list-1)
#
# for i in range(len(theory)):
#     print(int(theory[i]), int(s_list[i]))

# print(list(N_list))
# diff = [s_list[i+1] - s_list[i] for i in range(len(s_list)-1)]
# print(diff)

# n = 12
# results = []
# for k in range(1, n+1):
#     for m in range(1, k):
#         results.append((-1)**(m+k))
#
# print(results)
# print(sum(results))

#
# k = 15
# ss = 0
# for i in range(1, k):
#     ss += (-1)**i
#
# print(ss, -1/2 * (1-(-1)**(k-1)))


# N_list = np.arange(1, 100)
# theory = N_list/4 - 1/8 - ((-1)**(N_list-1))/8
# s_list = []
# for N in N_list:
#     s = 0
#     for n in range(1, N):
#         for k in range(1, n+1):
#             for m in range(1, n+1):
#                 s += (1/2)*(-1)**(m+k)
#     s_list.append(s)
#
# for i in range(len(theory)):
#     print(int(theory[i]) == int(s_list[i]))

# N = 10
# for k in range(1, N):
#     for m in range(k, N):
#         if 2*k-1 == 2*m:
#             print(True)
#         elif 2*k == 2*m-1:
#             print(True)
#         else:
#             continue

# N = 110
# result_1 = 0
# for k in range(1, N):
#     for n in range(k, N):
#         for m in range(1, n+1):
#             result_1 += 1
#
# result_2 = 0
#
# for k in range(1, N):
#     for m in range(k+1, N):
#         result_2 += N-m
#
# for k in range(1, N):
#     for m in range(1, k+1):
#         result_2 += N-k
#
# print(result_1, result_2)

# N = 150
#
# result_1 = 0
#
# for k in range(1, N):
#     for m in range(1, k+1):
#         result_1 += 1
#
# result_2 = 0
#
# for m in range(1, N):
#     for k in range(m+1, N):
#         result_2 += 1
#
# for m in range(1, N):
#     result_2 += 1
#
# print(result_1, result_2)

# N = 10
#
# for m in range(1, N):
#     for k in range(m, N):
#         if 2*k-1 == 2*m-1 or 2*k-1 == 2*m or 2*k == 2*m-1 or 2*k == 2*m:
#             print(True)






