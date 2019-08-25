import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import fenics as F
import dolfin
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


def residual(u_n, u_nm1, u_nm2, f_n, dt, c):
    '''
    Compute the residual of the PDE at time step n.
    '''
    # Project into higher-order functional space
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

    # Or use the space of same degree
    # f_n_ = F.project(f_n, V).vector()[:]
    # u_n_ = F.project(u_n, V).vector()[:]
    # u_nm1_ = F.project(u_nm1, V).vector()[:]
    # u_nm2_ = F.project(u_nm2, V).vector()[:]
    # ddu_n = F.project(u_n.dx(0).dx(0), V).vector()[:]

    R = np.sum(u_n_v - (dt ** 2) * (c ** 2) * ddu_n_v - (dt ** 2) * f_n_v -
               2 * u_nm1_v + u_nm2_v)
    return R


def pde_solver(f, u_D, u_I, c,
               f_boundary, divisions,
               T, dt, num_steps,
               initial_method='project', degree=1,
               u_e=None,
               backward=False):
    '''
    Solve

        d^2u/dt^2 = c^2 d^2u/dx^2 + f(x, t)

        s.t.    u(t) = u_D      @ DirichletBC
                du/dn = 0       @ NeumannBC
                u(x, 0) = u_I
                du(x, 0)/dt = 0

    with Backward Euler in time.

    Parameters
    ----------
    f: FEniCS.Expression
        Constitute linear form L(v), which is on the right hand side of
        the variational form. Could be time-dependent.
    u_D: List[FEniCS.Expression]
        Expressions for Dirichlet boundary conditions.
        Could be time-dependent.
    u_I: FEniCS.Expression
        Expression for initial condition.
    c: float
        Speed of sound in the medium defined by the problem.
    f_boundary: List[function]
        Functions that returns boolean value indicating on boundary or not.
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
    backward: bool
        Solve back in time or not. (TODO)

    Returns
    -------
    uL: np.array[float], shape (num_steps,)
        u(t) at the rightmost boundary.
    U: np.array[float], shape (num_steps, Nx)
        u(x, t) velocity field.
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
        a = F.inner(u, v) * F.dx +\
            (DT ** 2) * (C ** 2) * F.inner(F.grad(u), F.grad(v)) * F.dx
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
        a_M = F.inner(u, v) * F.dx
        a_K = F.inner(F.grad(u), F.grad(v)) * F.dx

        M = F.assemble(a_M)
        K = F.assemble(a_K)

        A = M + (dt ** 2) * (c ** 2) * K
        return M, K, A

    # Create mesh
    mesh = UnitHyperCube(divisions)

    # Define function spaces
    V = F.FunctionSpace(mesh, 'P', degree)

    # Define boundary conditions
    assert len(u_D) == len(f_boundary),\
        'Mismatch of the number of boundary expressions and boundary functions'
    bcs = []
    for ud, fb in zip(u_D, f_boundary):
        bcs.append(F.DirichletBC(V, ud[0], fb))

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

    U = []  # u(x, t) @ domain for outer iteration K
    uL = []  # u(t) @ boundary L
    xL = mesh.coordinates()[-1]  # coordinates @ boundary L
    t = dt
    fig = plt.figure()
    for n in range(num_steps):

        # Update current time
        t += dt

        # Update current terms
        idx = int(round(t * (num_steps / T)))
        if idx < 0:
            idx = 0
        if idx > num_steps - 1:
            idx = num_steps - 1
        print(idx)

        f.t_idx = idx

        bcs = []
        for ud, fb in zip(u_D, f_boundary):
            ud.idx = idx
            bcs.append(F.DirichletBC(V, ud[0], fb))

        # Solve
        # F.solve(a == L, u_,  bcu)

        # or equivalently
        # A = F.assemble(a)
        # b = F.assemble(L)
        # [bc.apply(A, b) for bc in bcs]
        # F.solve(A, u_.vector(), b)

        # or
        f_n = F.project(f[0], V).vector()
        b = 2 * M * u_nm1.vector() - M * u_nm2.vector() + (dt ** 2) * M * f_n
        [bc.apply(A, b) for bc in bcs]
        F.solve(A, u_.vector(), b)

        # Save solution to file (XDMF/HDF5)
        # xdmffile_u.write(u_, t)
        # Save nodal values to file
        # timeseries_u.store(u_.vector(), t)

        # Plot solution
        # plt.subplot(121)
        # F.plot(u_)
        # vertex_values = u_.compute_vertex_values(mesh)
        # plt.plot(np.linspace(mesh.coordinates().min(),
                             # mesh.coordinates().max(), len(vertex_values)),
                 # vertex_values)
        # plt.xlabel('x')
        # plt.ylabel('u')

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
            R = residual(u_, u_nm1, u_nm2, f[0], dt, c)
            print('[{:.2f}/{:.2f}]  R = {:.3g}'.format(t, T, R))

        # Update previous solution
        u_nm2.assign(u_nm1)
        u_nm1.assign(u_)

        # Save boundary L value @ t
        uL.append(u_(xL))
        U.append(F.interpolate(u_, V).vector()[:])

    uL = np.array(uL)
    U = np.array(U)

    ax = fig.add_subplot(111, projection='3d')

    X = np.linspace(mesh.coordinates().min(), mesh.coordinates().max(),
                    U.shape[1])
    T = np.linspace(0, T, U.shape[0])
    XX, TT = np.meshgrid(X, T)

    surf = ax.plot_surface(XX, TT, U, cmap='coolwarm')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')

    # plt.show()
    # plt.savefig('')
    return uL, U


def pde_solver_backward(f, boundary_conditions, u_I, c,
                        divisions, T, dt, num_steps,
                        initial_method='project', degree=1,
                        u_e=None):
    '''
    Solve

        d^2u/dt^2 = c^2 d^2u/dx^2 + f(x, t)

        s.t.    du/dn = u_N     @ NeumannBC
                u(x, T) = u_I
                du(x, 0)/dt = 0

    with Backward Euler in time.

    Parameters
    ----------
    f: FEniCS.Expression
        Constitute linear form L(v), which is on the right hand side of
        the variational form. Could be time-dependent.
    boundary_conditions: dict{int: {string: FEniCS.Expression}}
        {boundary_marker_id: {'boundary condition name': expression}}
        Boundary condition name could be: 'Dirichlet', 'Neumann' or 'Robin'.
        {'subboundary': FEniCS.SubDomain object}
    u_I: FEniCS.Expression
        Expression for initial condition.
    c: float
        Speed of sound in the medium defined by the problem.
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
    U: np.array[float], shape (num_steps, Nx)
        u(x, t) velocity field.
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

        # Collect Neumann conditions
        ds = F.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
        integrals_N = []
        for i in boundary_conditions:
            if isinstance(boundary_conditions[i], dict):
                if 'Neumann' in boundary_conditions[i]:
                    if boundary_conditions[i]['Neumann'] != 0:
                        g = boundary_conditions[i]['Neumann']
                        integrals_N.append(g[0] * v * ds(i))

        # Define variational problem
        a = F.inner(u, v) * F.dx +\
            (DT ** 2) * (C ** 2) * F.inner(F.grad(u), F.grad(v)) * F.dx
        L = ((DT ** 2) * f[0] + 2 * u_nm1 - u_nm2) * v * F.dx +\
            (DT ** 2) * (C ** 2) * sum(integrals_N)
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
        a_M = F.inner(u, v) * F.dx
        a_K = F.inner(F.grad(u), F.grad(v)) * F.dx

        M = F.assemble(a_M)
        K = F.assemble(a_K)

        A = M + (dt ** 2) * (c ** 2) * K

        ds = F.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
        M_ = F.assemble(u * v * ds(1))
        return M, K, A, M_

    # Create mesh
    mesh = UnitHyperCube(divisions)

    # Define function spaces
    V = F.FunctionSpace(mesh, 'P', degree)

    # Define boundary conditions
    boundary_markers = F.MeshFunction('size_t', mesh, 0)  # facet function
    subboundary_L = boundary_conditions['subboundary']
    subboundary_L.mark(boundary_markers, 1)  # mark boundary @ L

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
    M, K, A, M_ = _assembly(V)

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

    U = []  # u(x, t) @ domain for outer iteration K
    t = T - dt
    fig = plt.figure()
    for n in range(num_steps):

        # Update current time
        t -= dt

        # Update current terms
        idx = int(round(t * (num_steps / T)))
        if idx < 0:
            idx = 0
        if idx > num_steps - 1:
            idx = num_steps - 1
        print(idx)

        f.t_idx = idx

        for i in boundary_conditions:
            if isinstance(boundary_conditions[i], dict):
                if 'Neumann' in boundary_conditions[i]:
                    if boundary_conditions[i]['Neumann'] != 0:
                        boundary_conditions[i]['Neumann'].idx = idx

        g = boundary_conditions[1]['Neumann']
        print('g idx: ', g.idx)

        # Solve
        # F.solve(a == L, u_)

        # or
        f_n = F.project(f[0], V).vector()
        g_n = F.project(g[0], V).vector()
        b = 2 * M * u_nm1.vector() - M * u_nm2.vector() + (dt ** 2) * M * f_n +\
            (dt ** 2) * (c ** 2) * M_ * g_n
        # [bc.apply(A, b) for bc in bcs]
        F.solve(A, u_.vector(), b)

        # Save solution to file (XDMF/HDF5)
        # xdmffile_u.write(u_, t)
        # Save nodal values to file
        # timeseries_u.store(u_.vector(), t)

        # Plot solution
        # plt.subplot(121)
        # F.plot(u_)
        # vertex_values = u_.compute_vertex_values(mesh)
        # plt.plot(np.linspace(mesh.coordinates().min(),
                             # mesh.coordinates().max(), len(vertex_values)),
                 # vertex_values)
        # plt.xlabel('x')
        # plt.ylabel('z')

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
            R = residual(u_, u_nm1, u_nm2, f[0], dt, c)
            print('[{:.2f}/{:.2f}]  R = {:.3g}'.format(t, T, R))

        # Update previous solution
        u_nm2.assign(u_nm1)
        u_nm1.assign(u_)

        # Save boundary L value @ t
        U.append(F.interpolate(u_, V).vector()[:])

    U = np.array(U)

    ax = fig.add_subplot(111, projection='3d')

    X = np.linspace(mesh.coordinates().min(), mesh.coordinates().max(),
                    U.shape[1])
    T = np.linspace(0, T, U.shape[0])
    XX, TT = np.meshgrid(X, T)

    surf = ax.plot_surface(XX, TT, U[::-1, ...], cmap='coolwarm')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('z')

    plt.show()
    return U


def vocal_tract_solver(f_data, u0, uL, c_sound,
                       length, Nx, basis_degree,
                       T, num_tsteps):
    # f expression
    f_cppcode = """
    #include <iostream>
    #include <cmath>
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    #include <dolfin/function/Expression.h>

    class FExpress: public dolfin::Expression {
      public:
        int t_idx;  // time index
        double DX;  // length of uniformly spaced cell
        Eigen::MatrixXd array;  // external data

        // Constructor
        FExpress() : dolfin::Expression(1), t_idx(0) { }

        // Overload: evaluate at given point in given cell
        void eval(Eigen::Ref<Eigen::VectorXd> values,
                  Eigen::Ref<const Eigen::VectorXd> x) const {
          // values: values at the point
          // x: coordinates of the point
          int x_idx = std::round(x(0) / DX);  // spatial index
          values(0) = array(t_idx, x_idx);
        }
    };

    // Binding FExpress
    PYBIND11_MODULE(SIGNATURE, m) {
      pybind11::class_<FExpress, std::shared_ptr<FExpress>, dolfin::Expression>
      (m, "FExpress")
      .def(pybind11::init<>())
      .def_readwrite("t_idx", &FExpress::t_idx)
      .def_readwrite("DX", &FExpress::DX)
      .def_readwrite("array", &FExpress::array)
      ;
    }
    """
    f_expr = dolfin.compile_cpp_code(f_cppcode).FExpress()
    f = dolfin.CompiledExpression(f_expr, degree=basis_degree+2)
    f.array = f_data
    f.t_idx = 0
    f.DX = 1. / (Nx * basis_degree)

    # Initial expression
    u_I = F.Constant(0.)

    # Dirichlet boundary expression
    g_cppcode = """
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    #include <dolfin/function/Expression.h>

    class GExpress: public dolfin::Expression {
      public:
        int idx;  // time-dependent index
        Eigen::VectorXd array;  // external data

        // Constructor
        GExpress() : dolfin::Expression(1), idx(0) { }

        // Overload: evaluate at given point in given cell
        void eval(Eigen::Ref<Eigen::VectorXd> values,
                  Eigen::Ref<const Eigen::VectorXd> x) const {
          // values: values at the point
          // x: coordinates of the point
          values(0) = array(idx);
        }
    };

    // Binding GExpress
    PYBIND11_MODULE(SIGNATURE, m) {
      pybind11::class_<GExpress, std::shared_ptr<GExpress>, dolfin::Expression>
      (m, "GExpress")
      .def(pybind11::init<>())
      .def_readwrite("idx", &GExpress::idx)
      .def_readwrite("array", &GExpress::array)
      ;
    }
    """
    g_expr = dolfin.compile_cpp_code(g_cppcode).GExpress()

    u_D_0 = dolfin.CompiledExpression(g_expr, degree=basis_degree+2)
    u_D_0.array = u0
    u_D_0.idx = 0

    u_D_L = dolfin.CompiledExpression(g_expr, degree=basis_degree+2)
    u_D_L.array = uL
    u_D_L.idx = 0

    u_D = [u_D_0, u_D_L]

    def boundary_0(x, on_boundary):
        '''
        Test if x is on boundary.
        '''
        tol = 1E-14
        return on_boundary and (F.near(x[0], 0., tol))

    def boundary_L(x, on_boundary):
        '''
        Test if x is on boundary.
        '''
        tol = 1E-14
        return on_boundary and (F.near(x[0], length, tol))

    bcs = [boundary_0, boundary_L]

    # Solve
    dt = T / num_tsteps  # time step size
    uL_k, U_k = pde_solver(f, u_D, u_I, c_sound,
                           bcs, (Nx,),
                           T, dt, num_tsteps,
                           initial_method='project', degree=basis_degree)
    return uL_k, U_k


def vocal_tract_solver_backward(f_data, g_data, c_sound,
                                length, Nx, basis_degree,
                                T, num_tsteps):
    '''
    TODO
    '''
    # f expression
    f_cppcode = """
    #include <iostream>
    #include <cmath>
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    #include <dolfin/function/Expression.h>

    class FExpress: public dolfin::Expression {
      public:
        int t_idx;  // time index
        double DX;  // length of uniformly spaced cell
        Eigen::MatrixXd array;  // external data

        // Constructor
        FExpress() : dolfin::Expression(1), t_idx(0) { }

        // Overload: evaluate at given point in given cell
        void eval(Eigen::Ref<Eigen::VectorXd> values,
                  Eigen::Ref<const Eigen::VectorXd> x) const {
          // values: values at the point
          // x: coordinates of the point
          int x_idx = std::round(x(0) / DX);  // spatial index
          values(0) = array(t_idx, x_idx);
        }
    };

    // Binding FExpress
    PYBIND11_MODULE(SIGNATURE, m) {
      pybind11::class_<FExpress, std::shared_ptr<FExpress>, dolfin::Expression>
      (m, "FExpress")
      .def(pybind11::init<>())
      .def_readwrite("t_idx", &FExpress::t_idx)
      .def_readwrite("DX", &FExpress::DX)
      .def_readwrite("array", &FExpress::array)
      ;
    }
    """
    f_expr = dolfin.compile_cpp_code(f_cppcode).FExpress()
    f = dolfin.CompiledExpression(f_expr, degree=basis_degree+2)
    f.array = f_data
    f.t_idx = 0
    f.DX = 1. / (Nx * basis_degree)

    # Initial expression
    u_I = F.Constant(0.)

    # Neumann boundary expression
    g_cppcode = """
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    #include <dolfin/function/Expression.h>

    class GExpress: public dolfin::Expression {
      public:
        int idx;  // time-dependent index
        Eigen::VectorXd array;  // external data

        // Constructor
        GExpress() : dolfin::Expression(1), idx(0) { }

        // Overload: evaluate at given point in given cell
        void eval(Eigen::Ref<Eigen::VectorXd> values,
                  Eigen::Ref<const Eigen::VectorXd> x) const {
          // values: values at the point
          // x: coordinates of the point
          values(0) = array(idx);
        }
    };

    // Binding GExpress
    PYBIND11_MODULE(SIGNATURE, m) {
      pybind11::class_<GExpress, std::shared_ptr<GExpress>, dolfin::Expression>
      (m, "GExpress")
      .def(pybind11::init<>())
      .def_readwrite("idx", &GExpress::idx)
      .def_readwrite("array", &GExpress::array)
      ;
    }
    """
    g_expr = dolfin.compile_cpp_code(g_cppcode).GExpress()
    g = dolfin.CompiledExpression(g_expr, degree=basis_degree+2)
    g.array = g_data

    subboundary_L = F.CompiledSubDomain(
        'on_boundary && near(x[0], length, tol)',
        length=length, tol=1E-14)  # boundary @ L

    # Solve
    boundary_conditions = {1: {'Neumann': g},
                           'subboundary': subboundary_L}
    divisions = (Nx,)
    dt = T / num_tsteps  # time step size
    U_k = pde_solver_backward(f, boundary_conditions, u_I, c_sound,
                              divisions, T, dt, num_tsteps,
                              initial_method='project', degree=basis_degree)
    return U_k


if __name__ == '__main__':
    pass
