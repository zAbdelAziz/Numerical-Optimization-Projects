import numpy as np

from optimizer import Optimizer
from problems import PartI, PartII, PartIII, PartIV

# # Part I
# part_i = PartI()
# for i, func in enumerate(part_i.functions):
#     opt = Optimizer(func, part_i.grads[i])
#     print(opt.steepest_descent(part_i.x0[i], part_i.x_opt[i], alpha=part_i.alphas[i]))
# print('\n')
# for i, func in enumerate(part_i.functions):
#     opt = Optimizer(func)
#     print(opt.newton(part_i.x0[i], part_i.x_opt[i]))
# print('\n')
#
# # Part II
# part_ii = PartII()
# for i, x0 in enumerate(part_ii.x0s):
#     f, rjs, b = part_ii.least_squares(part_ii.qs[i], part_ii.ms[i], part_ii.ns[i])
#     opt = Optimizer(f)
#     # print(a)
#     print(opt.steepest_descent(x0, x_opt=0))
# print('\n')
# for i, x0 in enumerate(part_ii.x0s):
#     f, rjs, b = part_ii.least_squares(part_ii.qs[i], part_ii.ms[i], part_ii.ns[i])
#     opt = Optimizer(f)
#     print(opt.newton(x0, x_opt=0))
# print('\n')
#
# # Part III
# for n in [5, 8, 12, 20, 30]:
#     part_iii = PartIII(n)
#     opt = Optimizer(part_iii.f, grad=part_iii.grad_f)
#     x_opt = np.linalg.solve(part_iii.hilbert(n), np.ones(n))
#     print(x_opt)
#     # print(f(np.zeros(n), n))
#     print(opt.steepest_descent(part_iii.x0, x_opt=x_opt, alpha=100., rho=0.1, c=0.1))

# Part IV
part_iv = PartIV()
for i, func in enumerate(part_iv.functions):
    opt = Optimizer(func)
    print(opt.newton(part_iv.x_0, part_iv.x_opt[i], alpha=1, c=0.1))