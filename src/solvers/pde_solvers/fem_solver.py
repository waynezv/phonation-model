"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

import numpy as np
import sympy
from matplotlib import pyplot as plt
from tqdm import tqdm
import fenics as F
import mshr
import pdb


def UnitHyperCube(divisions):
    '''
    Make unit hyper-cube mesh with uniformly spaced cells.
    Currently only support maximum 3D.

    Parameters
    ----------
    divisions: List[int]
        List of number of vertices along each dimension.

    Returns
    -------
    mesh
    '''
    mesh_classes = [F.UnitIntervalMesh, F.UnitSquareMesh, F.UnitCubeMesh]
    d = len(divisions)
    mesh = mesh_classes[d - 1](*divisions)
    return mesh


def error_function(u_e, u, norm='L2'):
    '''
    Compute the error between estimated solution u and exact solution u_e
    in the specified norm.
    '''
    # Get function space
    V = u.function_space()

    if norm == 'L2':  # L2 norm
        # Explicit computation of L2 norm
        # E = ((u_e - u) ** 2) * F.dx
        # error = np.sqrt(np.abs(F.assemble(E)))

        # Implicit interpolation of u_e to higher-order elements.
        # u will also be interpolated to the space Ve before integration
        error = F.errornorm(u_e, u, norm_type='L2', degree_rise=3)

    elif norm == 'H10':  # H^1_0 seminorm
        error = F.errornorm(u_e, u, norm_type='H10', degree_rise=3)

    elif norm == 'max':  # max/infinity norm
        u_e_ = F.interpolate(u_e, V)
        error = np.abs(
            np.array(u_e_.vector()) - np.array(u.vector())
        ).max()

    return error


def pde_solver(f, u_D, u_I, c,
               f_boundary, divisions,
               T, dt, num_steps,
               initial_method='project', degree=1,
               u_e=None):
    '''
    Solve

        d^u/dt^2 = c^2 d^u/dx^2 + f(x, t)
        s.t.    u(t) = u_D  @ DirichletBC
                du/dn = 0   @ NeumannBC
                u(x, 0) = u_I
                du(x, 0)/dt = 0

    with Backward Euler in time.

    Parameters
    ----------
    f: FEniCS.Expression
        Constitute linear form L(v), which is on the right hand side of
        the variational form. Could be time-dependent.
    u_D: FEniCS.Expression
        Expression for Dirichlet boundary condition.
        Could be time-dependent.
    u_I: FEniCS.Expression
        Expression for initial condition.
    c: float
        Speed of sound in the medium defined by the problem.
    f_boundary: function
        Function that returns boolean value indicating on boundary or not.
    divisions: List[int]
        List of number of vertices along each dimension.
    T: float
        Total time span.
    dt: float
        Time step size.
    num_steps: int
        Number of time steps.
    initial_method: string
        Method to determine initial values.
        Can be either 'project' or 'interpolate'.
    degree: int
        Degree of the (Lagrange) polynomial element.
    u_e: FEniCS.Expression
        Exact solution.

    Returns
    -------
    '''
    def _make_variational_problem(V):
        '''
        Formulate the variational problem a(u, v) = L(v).

        Parameters
        ----------
        V: FEniCS.FunctionSpace

        Returns
        -------
        a, L: FEniCS.Expression
            Variational forms.
        '''
        # Define trial and test functions
        u = F.TrialFunction(V)
        v = F.TestFunction(V)

        # Define variational problem
        a = F.dot(u, v) * F.dx +\
            (DT ** 2) * (C ** 2) * F.dot(F.grad(u), F.grad(v)) * F.dx
        L = ((DT ** 2) * f + 2 * u_nm1 - u_nm2) * v * F.dx
        return a, L

    def _assembly(V):
        '''
        Assemble the matrices used in the linear system Au = b.
        Alternative representation of the variational form.
        Pre-calculate to speed up.
        '''
        # Define trial and test functions
        u = F.TrialFunction(V)
        v = F.TestFunction(V)

        # Assemble
        a_M = F.dot(u, v) * F.dx
        a_K = F.dot(F.grad(u), F.grad(v)) * F.dx

        M = F.assemble(a_M)
        K = F.assemble(a_K)

        A = M + (DT ** 2) * (C ** 2) * K
        return M, K, A

    def _residual(u_n, u_nm1, u_nm2, f_n):
        '''
        Compute the residual of the PDE at time step n.
        '''
        # BUG: project into higher order functional space?
        V = u_n.function_space()
        mesh = V.mesh()
        basis_degree = V.ufl_element().degree()
        U = F.FunctionSpace(mesh, 'P', basis_degree + 3)

        f_n_v = F.project(f_n, U).vector()[:]
        u_nm1_v = F.project(u_nm1, U).vector()[:]
        u_nm2_v = F.project(u_nm2, U).vector()[:]

        u_n_ = F.project(u_n, U)
        u_n_v = u_n_.vector()[:]
        ddu_n_v = F.project(u_n_.dx(0).dx(0), U).vector()[:]

        # or use the same order?
        # f_n_ = F.project(f_n, V).vector()[:]
        # u_n_ = F.project(u_n, V).vector()[:]
        # u_nm1_ = F.project(u_nm1, V).vector()[:]
        # u_nm2_ = F.project(u_nm2, V).vector()[:]
        # ddu_n = F.project(u_n.dx(0).dx(0), V).vector()[:]

        R = np.sum(u_n_v - (dt ** 2) * (c ** 2) * ddu_n_v - (dt ** 2) * f_n_v -
                   2 * u_nm1_v + u_nm2_v)
        return R

    # Create mesh
    mesh = UnitHyperCube(divisions)

    # Define function spaces
    V = F.FunctionSpace(mesh, 'P', degree)

    # Define boundary conditions
    bcu = F.DirichletBC(V, u_D, f_boundary)  # NOTE: vary with t

    # Define initial conditions
    # define functions for solutions at previous and current time steps
    if initial_method == 'project':
        # by u^0 = u_I
        # solving <u_nm2, v> = <u_I, v> for u_nm2
        u_nm2 = F.project(u_I, V)
        # by du^0/dt = 0 => u_nm1 = u_nm2
        # solving <u_nm1, v> = <u_nm2, v> for u_nm1
        u_nm1 = F.project(u_nm2, V)

    elif initial_method == 'interpolate':
        # by u^0 = U^j phi_j = u_I
        # setting U^j = u_I[j]
        u_nm2 = F.interpolate(u_I, V)
        # by du^0/dt = 0 => u_nm1 = U^j phi_j = u_nm2
        # setting U^j = u_nm2[j]
        u_nm1 = F.interpolate(u_nm2, V)

    # Define expressions used in variational forms
    DT = F.Constant(dt)  # NOTE: make a Constant(Expression)
    C = F.Constant(c)

    # Define variational forms
    # a, L = _make_variational_problem(V)

    # Or assemble linear systems
    M, K, A = _assembly(V)

    # Solve in time
    u_ = F.Function(V)  # NOTE: solution at current t, not trial function

    # TODO: log
    # F.set_log_level(F.LogLevel.PROGRESS)  # PROGRESS = 16

    # TODO: save?
    # Create XDMF files for visualization output
    # xdmffile_u = F.XDMFFile('vocal_tract_test/velocity.xdmf')
    # Create time series (for use in reaction_system.py)
    # timeseries_u = F.TimeSeries('vocal_tract_test/velocity_series')
    # Save mesh to file (for use in reaction_system.py)
    # F.File('vocal_tract_test/mesh.xml.gz') << mesh

    t = 0
    plt.figure()
    for n in tqdm(range(num_steps), desc='time stepping', leave=True):

        # Update current time
        t += dt
        f.t = t  # NOTE: necessary if bc vary with time
        u_D.t = t
        u_e.t = t

        # Solve
        # F.solve(a == L, u_,  bcu)

        # or equivalently
        # A = F.assemble(a)
        # b = F.assemble(L)
        # bcu.apply(A, b)
        # F.solve(A, u_.vector(), b)

        # or
        F_n = F.interpolate(f, V).vector()

        b = 2 * M * u_nm1.vector() - M * u_nm2.vector() + (DT ** 2) * M * F_n
        bcu.apply(A, b)
        F.solve(A, u_.vector(), b)

        # Save solution to file (XDMF/HDF5)
        # xdmffile_u.write(u_, t)
        # Save nodal values to file
        # timeseries_u.store(u_.vector(), t)

        # Plot solution
        plt.subplot(121)
        F.plot(u_)
        # vertex_values = u_.compute_vertex_values(mesh)
        # plt.plot(np.linspace(0, T, len(vertex_values)), vertex_values)
        plt.xlabel('x')
        plt.ylabel('u')

        # Compute error at vertices
        if u_e is not None:  # compute L2 error
            error = error_function(u_e, u_, norm='L2')
            print('[{:.2f}/{:.2f}]  error = {:.3g}'.format(t, T, error))

            u_e_ = F.interpolate(u_e, V)

            plt.subplot(122)
            F.plot(u_e_)
            plt.xlabel('x')
            plt.ylabel('u_e')

        else:  # compute residual
            R = _residual(u_, u_nm1, u_nm2, f)
            print('[{:.2f}/{:.2f}]  R = {:.3g}'.format(t, T, R))

        # Update previous solution
        u_nm2.assign(u_nm1)
        u_nm1.assign(u_)

    plt.show()


def toy_probelm_1():
    # Define some constants
    BASIS_DEGREE = 2  # degree of the basis functional space

    c = 1.  # speed of sound in medium
    alpha = 1.
    beta = 5.

    T = 2.0  # final time
    num_steps = 500  # number of time steps
    dt = T / num_steps  # time step size

    length = 1.  # spatial dimension
    divisions = (64,)  # mesh size

    f = F.Expression('2*beta - 2*alpha*c*c',
                     alpha=alpha, beta=beta, c=c, degree=BASIS_DEGREE+2)

    # Boundary expression
    g_expr = 'alpha*x[0]*x[0] + beta*t*t + 1'  # NOTE: x is a vector
    u_D = F.Expression(g_expr,  # NOTE: at initial time t=0
                       alpha=alpha, beta=beta, t=0, degree=BASIS_DEGREE+2)

    # Initial expression
    u_I = F.Expression(g_expr,
                       alpha=alpha, beta=beta, t=0, degree=BASIS_DEGREE+2)

    # Exact solution
    u_e = F.Expression(g_expr,
                       alpha=alpha, beta=beta, t=0, degree=BASIS_DEGREE+3)

    def boundary(x, on_boundary):
        '''
        Test if x is on boundary.
        '''
        tol = 1E-14
        return on_boundary and ((F.near(x[0], 0., tol)) or
                                (F.near(x[0], length, tol)))

    pde_solver(f, u_D, u_I, c,
               boundary, divisions,
               T, dt, num_steps,
               initial_method='project', degree=BASIS_DEGREE, u_e=u_e)


def toy_probelm_2():
    # Define some constants
    BASIS_DEGREE = 2  # degree of the basis functional space

    c = 1.  # speed of sound in medium
    alpha = 1.
    beta = 5.

    T = 2.0  # final time
    num_steps = 500  # number of time steps
    dt = T / num_steps  # time step size

    length = 1.  # spatial dimension
    divisions = (128,)  # mesh size

    f = F.Expression('2*beta + 2*alpha*x[0]*x[0] - 2*alpha*c*c*t*t',
                     alpha=alpha, beta=beta, c=c, t=0, degree=4)
    # f = F.Expression('-2*alpha*c*c*t',
                     # alpha=alpha, beta=beta, c=c, t=0, degree=BASIS_DEGREE+2)

    # Boundary expression
    g_expr = '(alpha*x[0]*x[0] + beta)*t*t'
    # g_expr = '(alpha*x[0]*x[0] + beta)*t'
    u_D = F.Expression(g_expr,
                       alpha=alpha, beta=beta, t=0, degree=BASIS_DEGREE+2)

    # Initial expression
    # u_I = F.Constant(0.)
    u_I = F.Expression(g_expr,
                       alpha=alpha, beta=beta, t=0, degree=BASIS_DEGREE+2)

    # Exact solution
    u_e = F.Expression(g_expr,
                       alpha=alpha, beta=beta, t=0, degree=BASIS_DEGREE+3)

    def boundary(x, on_boundary):
        '''
        Test if x is on boundary.
        '''
        tol = 1E-14
        return on_boundary and ((F.near(x[0], 0., tol)) or
                                (F.near(x[0], length, tol)))

    pde_solver(f, u_D, u_I, c,
               boundary, divisions,
               T, dt, num_steps,
               initial_method='project', degree=BASIS_DEGREE, u_e=u_e)


def test_solver():
    toy_probelm_1()
    toy_probelm_2()


#
# Use SymPy to compute f from the manufactured solution u
# x, t = sympy.symbols('x[0], x[1]')
# u = 1 + alpha * (x ** 2) + beta * (t ** 2)

# x_coord = np.linspace(0, length, 64 + 1)
# t_coord = np.linspace(0, T, num_steps)
# # xv, tv = np.meshgrid(x_coord, t_coord)
# # z = 1 + alpha * (xv ** 2) + beta * (tv ** 2)
# z = lambda x, t: 1 + alpha * (x ** 2) + beta * (t ** 2)
# plt.figure()
# # h = plt.contourf(x_coord, t_coord, z)
# for t in t_coord:
    # zv = list(map(z, x_coord, [t] * len(x_coord)))
    # plt.plot(x_coord, zv)
# plt.xlim([0, 1])
# plt.ylim([1, 2])
# plt.show()

# f = sympy.diff(sympy.diff(u, t), t) - (c ** 2) * sympy.diff(sympy.diff(u, x), x)
# f = sympy.simplify(f)
# u_0 = u.subs(x, 0)
# u_L = u.subs(x, length)

# u_code = sympy.printing.ccode(u)
# f_code = sympy.printing.ccode(f)
# u0_code = sympy.printing.ccode(u_0)
# uL_code = sympy.printing.ccode(u_L)
# print('u =', u_code)
# print('f =', f_code)
# print('u_0 =', u0_code)
# print('u_L =', uL_code)


if __name__ == '__main__':
    test_solver()
