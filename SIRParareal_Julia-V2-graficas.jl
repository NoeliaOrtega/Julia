using LaTeXStrings
# Parámetros

const global N_ind = 103 # Número de individuos
const global beta = 0.08 # Infección
const global nu = 0.75 # Recuperación

const global S0 = 100. # Número de susceptibles
const global I0 = 3. # Número de infectados
const global R0 = N_ind-S0-I0

const global t_init = 0.
const global t_end  = 5.
t_n_f = 100000
const global t_n_c = 100

const global N_intervalos = 160;
const global T = LinRange(t_init, t_end, N_intervalos+1)

const global n_iter = 10 # Número de iteraciones de Parareal

#Sistema
SIR_1(S,I) = -beta*S*I
SIR_2(S,I) = beta*S*I - nu*I
SIR_3(S,I) = nu*I

function SIR(U, t)
     S = U[1]
     I = U[2]
     return [-beta*S*I, beta*S*I-nu*I, nu*I]
end

# Resuelve el sistema de EDO U' = F(U,t), U(0)=U0 en el rango 
# de tiempo T = [t_0, ..., t_n] mediante el método de Euler
function Euler_SIR(U0, t_init, t_end, n_t)
    dt = (t_end-t_init)/n_t
    t = t_init
    
    #println("n_t=$n_t, t_init=$t_init, t_end=$t_end, dt=$dt")
    S0, I0, R0 = U0
    S1, I1, R1 = U0
    for i=1:n_t
        # Denotamos: U=solución en la etapa actual, U0=sol. etapa anterior
        S1 = S0 + dt*SIR_1(S0,I0)
        I1 = I0 + dt*SIR_2(S0,I0)
        R1 = R0 + dt*SIR_3(S0,I0)

        # Preparamos siguiente iteración
        t += dt
        S0, I0, R0 = S1, I1, R1
        
        #println("Iter $i, t=$t, u=$U_sol, ex_sol=$(exp(t))")

    end
    return [S1, I1, R1]
end

@inline F(t1, t0, u0) =  Euler_SIR( u0, t0, t1, t_n_f)
@inline G(t1, t0, u0) =  Euler_SIR(u0, t0, t1, t_n_c)

function SIR_secuencial()
    U = Array{Float64,3}(undef, N_intervalos+1, n_iter+1, 3);
    Fn = Array{Float64,2}(undef, N_intervalos+1, 3);

    # 1.a) Inicialización (aproximción grosera)
    U[1,1,:] = U0
    for n=1:N_intervalos
        U[n+1,1,:] = G( T[n+1],T[n],U[n,1,:] )
    end
        
    # 1.b) Inicialización etapas parareal
    for k=1:n_iter
        U[1,k+1,:] = U0
    end

    # 2) Bucle parareal
    for k=1:n_iter
 
        # 2.a) Aproximación fina (secuencial) en cada subintervalo
        @inbounds begin
        for n = 1:N_intervalos
            t0 = T[n]
            t1 = T[n+1]
            Fn[n,:] = F( t1, t0, U[n,k,:] )
        end
        end 
        
        # 2.b) Corrección secuencial
        @inbounds for n = 1:N_intervalos
            U[n+1, k+1, :] = Fn[n,:] + G( T[n+1], T[n], U[n,k+1,:] ) - G( T[n+1], T[n], U[n,k,:] )
        end
    end
    
    return U
    
end

function SIR_parareal()
    U = Array{Float64,3}(undef, N_intervalos+1, n_iter+1, 3);
    Fn = Array{Float64,2}(undef, N_intervalos+1, 3);

    # 1.a) Inicialización (aproximción grosera)
    U[1,1,:] = U0
    for n=1:N_intervalos
        U[n+1,1,:] = G( T[n+1],T[n],U[n,1,:] )
    end
        
    # 1.b) Inicialización etapas parareal
    @inbounds Threads.@threads for k=1:n_iter
        U[1,k+1,:] = U0
    end

    # 2) Bucle parareal
    for k=1:n_iter
 
        # 2.a) Aproximación fina (paralela) en cada subintervalo
        begin
        @inbounds Threads.@threads for n = 1:N_intervalos
            t0 = T[n]
            t1 = T[n+1]
            Fn[n,:] = F( t1, t0, U[n,k,:] )
        end 
        end
        
        # 2.b) Corrección secuencial
        @inbounds for n = 1:N_intervalos
            U[n+1, k+1, :] = Fn[n,:] + G( T[n+1], T[n], U[n,k+1,:] ) - G( T[n+1], T[n], U[n,k,:] )
        end
    end
    
    return U
    
end

@time U1 = SIR_secuencial();

@time U2 = SIR_parareal();

#import Pkg; Pkg.add("Plots")

using Plots

y = U2[:,end,:]; 
y1, y2, y3 = y[:,1], y[:,2], y[:,3]
x = T
plot(x, y, title = "Evolución de la epidemia", label = ["Susceptibles" "Infectados" "Recuperados"], lw = 2)

time_16= Array{Float64,2}(undef, 5,2);
for i=1:5
    t_n_f = 100000*i
    tiempo = @elapsed SIR_secuencial()
    time_16[i,1] = tiempo
    tiempo = @elapsed SIR_parareal()
    time_16[i,2] = tiempo
end

time_array=LinRange(1, 5, 5)

plotly()
plot(time_array,  time_16, label = ["secuencial" "paralelo"], title = "N_intervalos = núcleos", xlabel="iteraciones finas (x100000)", ylabel="tiempo (s)")

time_32= Array{Float64,2}(undef, 5,2);
for i=1:5
    t_n_f = 100000*i
    tiempo = @elapsed SIR_secuencial()
    time_32[i,1] = tiempo
    tiempo = @elapsed SIR_parareal()
    time_32[i,2] = tiempo
end

plotly()
plot(time_array,  time_32, label = ["secuencial" "paralelo"], title = "N_intervalos = 2*núcleos", xlabel="iteraciones finas (x100000)", ylabel="tiempo (s)")

time_8= Array{Float64,2}(undef, 5,2);
for i=1:5
    t_n_f = 100000*i
    tiempo = @elapsed SIR_secuencial()
    time_8[i,1] = tiempo
    tiempo = @elapsed SIR_parareal()
    time_8[i,2] = tiempo
end

plotly()
plot(time_array,  time_8, label = ["secuencial" "paralelo"], title = "N_intervalos = mitad de núcleos", xlabel="iteraciones finas (x100000)", ylabel="tiempo (s)")

time_160= Array{Float64,2}(undef, 5,2);
for i=1:5
    t_n_f = 100000*i
    tiempo = @elapsed SIR_secuencial()
    time_160[i,1] = tiempo
    tiempo = @elapsed SIR_parareal()
    time_160[i,2] = tiempo
end


plotly()
plot(time_array,  time_160, label = ["secuencial" "paralelo"], title = "N_intervalos = 10*núcleos", xlabel="iteraciones finas (x100000)", ylabel="tiempo (s)")


