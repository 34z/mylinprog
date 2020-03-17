import numpy as np
import scipy.optimize
from cz.data import *

def simplex(A, b, c, x, min_or_max='max', max_iter=1000):
	A, b, c, x = npfloatarray(A, b, c, x)
	base_index, = np.where(x != 0)
	cnt = np.linalg.matrix_rank(A) - len(base_index)
	if cnt:
		non_index, = np.where(x == 0)
		for i in range(cnt):
			base_index = np.append(base_index, non_index[i])
	infinite_flag = False
	iter = 0
	while True:
		A_b = A[:, base_index]
		c_b = c[base_index]
		A_b_I = np.linalg.inv(A_b)

		A_bar = A_b_I @ A
		b_bar = A_b_I @ b
		z = c_b.transpose() @ A_b_I @ b
		sigma = c.transpose() - c_b.transpose() @ A_b_I @ A

		if min_or_max == 'min':
			in_base_index, = np.where(sigma < -1e-8)
		else:
			in_base_index, = np.where(sigma > 1e-8)
		if len(in_base_index):
			for i in range(len(in_base_index)):
				in_base = in_base_index[i]
				# print("in", in_base)
				if np.alltrue(A_bar[:, in_base] <= 0):
					infinite_flag = True
					break
				else:
					pos_index, = np.where(A_bar[:, in_base] > 0)
					tmp = b_bar[pos_index] / A_bar[pos_index, in_base]
					out_base_idx = pos_index[np.argmin(tmp)]
					# print("out", base_index[out_base_idx])
					out_base = base_index[out_base_idx]
					base_index[out_base_idx] = in_base
					if np.linalg.det(A[:, base_index]) == 0:
						base_index[out_base_idx] = out_base
					else:
						break
			if infinite_flag:
				break
		else:
			break
		iter += 1
		if iter > max_iter:
			A_b = A[:, base_index]
			c_b = c[base_index]
			A_b_I = np.linalg.inv(A_b)
			b_bar = A_b_I @ b
			z = c_b.transpose() @ A_b_I @ b
			break

	if infinite_flag:
		print("Infinite BFS.")
		x, z = None, None
	else:
		x[:] = 0.
		x[base_index] = b_bar
	return x, z

def simplex_two_step(A, b, c, max_or_min):
	m, n = A.shape
	A1 = np.hstack((A, np.identity(m)))
	c1 = np.zeros(m + n)
	c1[n:] = 1.
	x = np.zeros(m + n)
	x[n:] = 1.

	x, z = simplex(A1, b, c1, x, 'min')
	if z == 0:
		x, z = simplex(A, b, c, x[:n], max_or_min)
	else:
		print("No BFS")
		x, z = None, None
	return x, z


if __name__ == '__main__':
	A = np.array([[1, -2, 1, 1, 0],
				  [-4, 1, 2, 0, -1],
				  [-2, 0, 1, 0, 0]])
	b = np.array([11, 3, 1])
	c = np.array([3, -1, -1, 0, 0])

	# A = np.array([[30, 20, 1, 0, 0],
	# 			  [5, 1, 0, 1, 0],
	# 			  [1, 0, 1, 0, 1]])
	# b = np.array([160, 15, 4])
	# c = np.array([5, 2, 0, 0, 0])

	# A = np.array([[1, 0, 0, 0.25, -8, -1, 9],
	# 			  [0, 1, 0, 0.5, -12, -0.5, 3],
	# 			  [0, 0, 1, 0, 0, 1, 0]])
	# b = np.array([0, 0, 1])
	# c = np.array([0, 0, 0, 0.75, -20, 0.5, -6])

	# A = np.array([[1, 1, 1, 1, 0, 0],
	# 			  [-1, 2, -2, 0, 1, 0],
	# 			  [2, 1, 0, 0, 0, 1]])
	# b = np.array([4, 6, 5])
	# c = np.array([1, 2, -1, 0, 0, 0])

	# A = np.array([[1, 1, 2, 1, 3, -1, 0],
	# 			  [2, -1, 3, 1, 1, 0, -1]])
	# b = np.array([4, 3])
	# c = np.array([2, 3, 5, 2, 3, 0, 0])

	x, z = simplex_two_step(A, b, c, 'max')
	print(x)
	print(z)
