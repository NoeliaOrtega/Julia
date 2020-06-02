Threads.nthreads()

using Gridap
import Gridap: ∇

#En primer lugar vamos a definir la expresión de la solución ya conocida

u_exacta(x) = (x[1]*x[1]-x[1])*(x[2]*x[2]-x[2])*t

#Mallado
domain = (0,1,0,1)
partition = (20,20)
model = CartesianDiscreteModel(domain,partition)
model=simplexify(model)
trian = Triangulation(model)
degree = 1
quad = CellQuadrature(trian,degree)

dt_ex = 1/N_intervalos

for i=1:N_intervalos+1
    t=(i-1)*dt_ex
    writevtk(trian,"Sol_exacta$(lpad(i,2,'0'))",cellfields=["u_exacta" => u_exacta])
end 

#Parámetros
t = 0.
g(x) = 0  #valor en la frontera
f(x) = (x[1]*x[1]-x[1])*(x[2]*x[2]-x[2]) - 2*(x[2]*x[2]-x[2])*t - (x[1]*x[1]-x[1])*2*t  #fuente de calor
beta = 1.
u0(x) = 0.  #valor inicial

t_init, t_end = 0., 1.
n_t = 10;
dt = (t_end - t_init)/n_t;
N_intervalos = 10;
n_iter = 8;
T = LinRange(t_init, t_end, N_intervalos+1)


#Espacio de funciones
order = 1
V0 = TestFESpace(
  reffe=:Lagrangian, order=order, valuetype=Float64,
  conformity=:H1, model=model, dirichlet_tags="boundary")

U = TrialFESpace(V0,g)


function EulerMEF(t_init,t_end,n_t,u0)
    dt = (t_end - t_init)/n_t
    u_sol = u0
    t = t_init
    f(x) = (x[1]*x[1]-x[1])*(x[2]*x[2]-x[2]) - 2*(x[2]*x[2]-x[2])*t - (x[1]*x[1]-x[1])*2*t
    
    a(u,v) = dt * beta*∇(v)*∇(u) + u*v
    b(v) = v*dt*f + v*u0
    
    for i=1:n_t
        t += dt
        t_Ω = AffineFETerm(a,b,trian,quad)
        op = AffineFEOperator(U,V0,t_Ω)
        u_sol = solve(op)
        
        u0 = u_sol
    end
    return u_sol
end 
    

t_n_f = 70;
t_n_c = 10;

F(t1, t0, u0) =  EulerMEF(t0, t1, t_n_f,u0)
G(t1, t0, u0) =  EulerMEF(t0, t1, t_n_c,u0)

#Función de Gridap con el valor inicial 
f2(x) = 0

U_u0 = TrialFESpace(V0,g)

a_u0(u,v) =  beta*∇(v)*∇(u) + u*v
b_u0(v) = v*f2 + v*u0

t_Ω_u0 = AffineFETerm(a_u0,b_u0,trian,quad)
op_u0 = AffineFEOperator(U_u0,V0,t_Ω_u0)

u0_guardar = solve(op_u0)

typeof(u0_guardar)

function EDP_secuencial()
    U = Array{Gridap.Geometry.GenericCellField{true},2}(undef, N_intervalos+1, n_iter+1);
    F_sol = Array{Gridap.Geometry.GenericCellField{true},1}(undef, N_intervalos+1);

    # 1.a) Inicialización (aproximción grosera)
    U[1,1] = 1*u0_guardar

    for n=1:N_intervalos
        U[n+1,1] = 1*G( T[n+1],T[n],U[n,1] )
    end
        
    # 1.b) Inicialización etapas parareal
    for k=1:n_iter
        U[1,k+1] = 1*u0_guardar
    end

    # 2) Bucle parareal
    for k=1:n_iter
 
        # 2.a) Aproximación fina (paralela) en cada subintervalo
        for n = 1:N_intervalos
            F_sol[n] = 1*F( T[n+1], T[n], U[n,k] )
        end
        
        # 2.b) Corrección secuencial
        for n = 1:N_intervalos
            U[n+1, k+1] = F_sol[n] + G( T[n+1], T[n], U[n,k+1] ) - G( T[n+1], T[n], U[n,k] )
        end
    end
    
    return U
    
end

function EDP_parareal()
    U = Array{Gridap.Geometry.GenericCellField{true},2}(undef, N_intervalos+1, n_iter+1);
    F_sol = Array{Gridap.Geometry.GenericCellField{true},1}(undef, N_intervalos+1);

    # 1.a) Inicialización (aproximción grosera)
    U[1,1] = 1*u0_guardar

    for n=1:N_intervalos
        U[n+1,1] = 1*G( T[n+1],T[n],U[n,1] )
    end
        
    # 1.b) Inicialización etapas parareal
    @inbounds Threads.@threads for k=1:n_iter
        U[1,k+1] = 1*u0_guardar
    end

    # 2) Bucle parareal
    for k=1:n_iter
 
        # 2.a) Aproximación fina (paralela) en cada subintervalo
        @inbounds Threads.@threads for n = 1:N_intervalos
            F_sol[n] = 1*F( T[n+1], T[n], U[n,k] )
        end
        
        # 2.b) Corrección secuencial
        for n = 1:N_intervalos
            U[n+1, k+1] = F_sol[n] + G( T[n+1], T[n], U[n,k+1] ) - G( T[n+1], T[n], U[n,k] )
        end
    end
    
    return U
    
end



@time U1 = EDP_secuencial();

@time U2 = EDP_parareal();

for i=1:N_intervalos+1
    sol = U2[i,end]
    writevtk(trian,"Sol_numerica$(lpad(i,2,'0'))",cellfields=["sol" => sol])
end 

for i=1:N_intervalos+1
    t=(i-1)*dt_ex
    e = u_exacta - U2[i,end]
    writevtk(trian,"Error_t$(lpad(i,2,'0'))",cellfields=["e" => e])
end 

# Error en norma L2 para cada subintervalo temporal

l2(w) = w*w


error_L2 = Array{Any,1}(undef, N_intervalos+1);

for i=1:N_intervalos+1
    el2 = sqrt(sum( integrate(l2(U2[i,end]-u_exacta),trian,quad) ))
    error_L2[i] = el2
end 

using Plots

plot(T,error_L2,
    shape=:auto,
    xlabel="T",ylabel="error L2")

U2

for i=1:n_iter+1
    t=1.
    e = u_exacta - U2[end,i]
    writevtk(trian,"Error_k$(lpad(i,2,'0'))",cellfields=["e" => e])
end 

# Error en norma L2 para cada subintervalo temporal

l2(w) = w*w


error_L2_k = Array{Any,1}(undef, n_iter+1);

for i=1:n_iter+1
    el2 = sqrt(sum( integrate(l2(U2[end,i]-u_exacta),trian,quad) ))
    error_L2_k[i] = el2
end 

K = LinRange(0, n_iter, n_iter+1)
plot(K,error_L2_k,
    shape=:auto,
    xlabel="iteración",ylabel="error L2")


