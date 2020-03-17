import numpy as np
from cz.data import *


def _to_canonical_form(r_objective, r_constrain, basis):
	for i in range(len(basis)):
		r_constrain[i] /= r_constrain[i, basis[i]]
		for row in range(r_constrain.shape[0]):
			if row != i:
				r_constrain[row] += -r_constrain[i] * r_constrain[row, basis[i]]
		r_objective += -r_constrain[i] * r_objective[basis[i]]
	return r_objective, r_constrain


def _apply_base_change(r_objective, r_constrain, basis, in_base, out_base_idx):
	basis[out_base_idx] = in_base
	r_constrain[out_base_idx, :] /= r_constrain[out_base_idx, in_base]
	for row in range(r_constrain.shape[0]):
		if row != out_base_idx:
			r_constrain[row] += -r_constrain[out_base_idx] * r_constrain[row, basis[out_base_idx]]
	r_objective += -r_constrain[out_base_idx] * r_objective[basis[out_base_idx]]


def _solve_simplex(r_objective, r_constrain, basis, tol=1E-12, max_iter=1000):
	n = r_constrain.shape[1] - 1
	n_iter = 0
	status = 0
	quit = False
	_to_canonical_form(r_objective, r_constrain, basis)
	while not quit:
		in_bases = [var for var in range(n) if r_objective[var] < -tol]
		if len(in_bases) == 0:
			# best solution
			status = 0
			quit = True
		else:
			in_base = in_bases[0]
			out_base_mask = np.less(r_constrain[:, in_base], tol)  # TODO: 0 or tol ?
			if np.alltrue(out_base_mask):
				# unbound
				status = 1
				quit = True
			else:
				out_base_mask = ~out_base_mask
				out_base_idx = np.argmin(r_constrain[out_base_mask, -1] / r_constrain[out_base_mask, in_base])
				out_base_idx = np.arange(len(basis))[out_base_mask][out_base_idx]
				if n_iter < max_iter:
					_apply_base_change(r_objective, r_constrain, basis, in_base, out_base_idx)
					# basis[out_base_idx] = in_base
					n_iter += 1
				else:
					# over max iteration
					status = 2
					quit = True
	solution = np.zeros(n)
	solution[basis] = r_constrain[:, -1]
	opt = -r_objective[-1]
	return solution, opt, status


def simplex_eq(c, A_eq, b_eq, tol=1E-12, max_iter=1000):
	A_eq, b_eq, c = npfloatarray(A_eq, b_eq, c)
	# A_eq: m constrains * n variables
	m, n = A_eq.shape

	# let b_eq >= 0 for phase 1
	neg_constrains = np.less(b_eq, 0)
	A_eq[neg_constrains] *= -1
	b_eq[neg_constrains] *= -1

	# create m artificial variables
	var_artificial = np.arange(m) + n

	# phase 1
	r_constrain = np.hstack((A_eq, np.eye(m), b_eq[:, np.newaxis]))
	r_objective= np.hstack((np.zeros(n), np.ones(m), 0))
	basis = var_artificial.copy()
	solution, opt, status = _solve_simplex(r_objective, r_constrain, basis,
										   tol=tol, max_iter=max_iter)

	# phase 2
	if abs(opt) < tol:
		r_objective = np.hstack((c, 0))
		r_constrain = np.delete(r_constrain, var_artificial, axis=1)
		solution, opt, status = _solve_simplex(r_objective, r_constrain, basis,
											   tol=tol, max_iter=max_iter)
	else:
		# Failure to find a feasible starting point
		status = 3
		for out_base_idx in [row for row in range(m)
					   if basis[row] > n - 1]:
			non_zero_row = [col for col in range(n)
							if abs(r_constrain[out_base_idx, col]) > tol]
			if len(non_zero_row) > 0:
				in_base = non_zero_row[0]
				_apply_base_change(r_objective, r_constrain, basis, in_base, out_base_idx)
		r_objective = np.hstack((c, 0))
		r_constrain = np.delete(r_constrain, var_artificial, axis=1)
		_to_canonical_form(r_objective, r_constrain, basis)
		solution = np.zeros(n)
		solution[basis] = r_constrain[:, -1]
		opt = -r_objective[-1]

	return solution, opt, status


def simplex(c, A_eq, b_eq, A_le, b_le, tol=1E-12, max_iter=1000):
	m_eq, n_eq = A_eq.shape
	m_le, n_le = A_le.shape
	n_slack = m_le
	A_eq = np.hstack((A_eq, np.zeros((m_eq, n_slack))))
	A_le = np.hstack((A_le, np.eye(n_slack)))
	A = np.vstack((A_le, A_eq))
	b = np.vstack((b_le[:, np.newaxis], b_eq[:, np.newaxis])).ravel()

	solution, opt, status = simplex_eq(c, A, b, tol=tol, max_iter=max_iter)

	return solution[:-n_slack], opt, status


if __name__ == '__main__':
	A = np.array([[1, -2, 1, 1, 0],
				  [-4, 1, 2, 0, -1],
				  [-2, 0, 1, 0, 0]])
	b = np.array([11, 3, 1])
	c = np.array([-3., 1, 1, 0, 0])

	A_le = np.array([[1, -2, 1],
				  [4, -1, -2]])
	b_le = np.array([11, -3])
	A_eq = np.array([[-2, 0, 1]])
	b_eq = np.array([1])
	c = np.array([-3, 1, 1, 0, 0])

	# A = np.array([[30, 20, 1, 0, 0],
	# 			  [5, 1, 0, 1, 0],
	# 			  [1, 0, 1, 0, 1]])
	# b = np.array([160, 15, 4])
	# c = np.array([5, 2, 0, 0, 0])

	# A = np.array([[1, 0, 0, 0.25, -8, -1, 9],
	# 			  [0, 1, 0, 0.5, -12, -0.5, 3],
	# 			  [0, 0, 1, 0, 0, 1, 0]])
	# b = np.array([0, 0, 1])
	# c = np.array([0, 0, 0, -0.75, 20, -0.5, 6])

	# A = np.array([[1, 1, 1, 1, 0, 0],
	# 			  [-1, 2, -2, 0, 1, 0],
	# 			  [2, 1, 0, 0, 0, 1]])
	# b = np.array([4, 6, 5])
	# c = np.array([1, 2, -1, 0, 0, 0])

	# A = np.array([[1, 1, 2, 1, 3, -1, 0],
	# 			  [2, -1, 3, 1, 1, 0, -1]])
	# b = np.array([4, 3])
	# c = np.array([2, 3, 5, 2, 3, 0, 0])

	# solution, opt, status = simplex_eq(c, A, b)
	# if status == 0:
	# 	print(solution)
	# 	print(opt)
	solution, opt, status = simplex(c, A_eq, b_eq, A_le, b_le)
	import scipy.optimize
	scipy.optimize.linprog()
	if status == 0:
		print(solution)
		print(opt)
