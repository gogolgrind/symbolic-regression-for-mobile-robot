from acados_template import AcadosModel
import casadi as cd

def robot_model():
    model = AcadosModel()

    model.name = "robot_model"

    # State
    x = cd.SX.sym('x') 
    y = cd.SX.sym('y')   
    v = cd.SX.sym('v')  
    theta = cd.SX.sym('theta') 

    sym_x = cd.vertcat(x, y, v ,theta)
    model.x = sym_x

    # Input
    a = cd.SX.sym('a')
    w = cd.SX.sym('w')

    sym_u = cd.vertcat(a, w)
    model.u = sym_u

    # Derivative of the States
    x_dot = cd.SX.sym('x_dot')
    y_dot = cd.SX.sym('y_dot')
    theta_dot = cd.SX.sym('theta_dot')
    v_dot = cd.SX.sym('v_dot')
    x_dot = cd.vertcat(x_dot, y_dot, v_dot, theta_dot)

    model.xdot = x_dot
    

    ## Model of Robot
    f_expl = cd.vertcat(sym_x[2] * cd.cos(sym_x[3]),
                    sym_x[2] * cd.sin(sym_x[3]),
                    sym_u[0],
                    sym_u[1])
    f_impl = x_dot - f_expl
 
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl


    num_static_obst = 40
    obst_param = cd.SX.sym('p', 2 * num_static_obst)
    sym_p = cd.vertcat(obst_param)

     #model.p = sym_p

    x_obst = obst_param[0::2]
    y_obst = obst_param[1::2]


    # potential function for a static obstacles
    obst_stat = cd.SX.sym('obst_stat',1)
    e = 6
    #for i in range(num_static_obst): 
    # заменить на другую формулу
    # distance = (2.5-x)**2/0.5**2 + (2.8-y)**2 / 0.5**2
    # w2 = 10
        
    
    obst_stat = 0.00455 + 9.98/(319488833.70232*(0.69137*(0.39855*x - 1)**2 + (0.39696*y - 1)**2 + 0.0002)**6 + 0.663)
    # 5*(cd.pi/2 + cd.atan(w2 - distance*w2)) 0.0029055786*cd.exp((y**2)*cd.atan(x*(0.035857655*cd.exp(x**2)+0.00777126856727905)))+0.012730344

    model.cost_y_expr = cd.vertcat(sym_x, sym_u, obst_stat)
    model.cost_y_expr_e = cd.vertcat(sym_x, obst_stat)  
  

    return model