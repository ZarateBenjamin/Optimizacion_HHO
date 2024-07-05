"1er intento zarate de implementar el algoritmo HHO"


import numpy as np

# Parámetros de los anuncios con: [clientes potenciales, costo min, costo max, valorización min, valorización max, max anuncios]
anuncios = {
    'TV_tarde': [1000, 160, 200, 65, 85, 15],
    'TV_noche': [2000, 300, 350, 90, 95, 10],
    'Diarios': [1500, 40, 80, 40, 60, 25],
    'Revistas': [2500, 100, 120, 60, 80, 4],
    'Radio': [300, 10, 20, 20, 30, 30]
}

def calcular_costo(calidades):
    costo_total = 0
    for i, key in enumerate(anuncios):
        rango_valor = anuncios[key][4] - anuncios[key][3]
        rango_costo = anuncios[key][2] - anuncios[key][1]
        proporción = (calidades[i] - anuncios[key][3]) / rango_valor if rango_valor else 0
        costo = anuncios[key][1] + rango_costo * proporción
        costo_total += costo
    return costo_total

def calcular_calidad(calidades):
    return np.sum(calidades)

def es_solucion_valida(calidades):
    costos = calcular_costo(calidades)
    # Añadir restricciones presupuestarias específicas si es necesario
    return np.all(calidades >= 0) and np.all(calidades <= [anuncio[5] for anuncio in anuncios.values()]) and costos <= max_cost

def hho(N, T):
    # Inicializar la población
    dimension = len(anuncios)
    X = np.random.uniform(low=[anuncio[3] for anuncio in anuncios.values()],
                          high=[anuncio[4] for anuncio in anuncios.values()], size=(N, dimension))
    fitness = np.array([calcular_calidad(xi) - calcular_costo(xi) for xi in X])
    Xrabbit = X[np.argmax(fitness)]
    best_fitness = np.max(fitness)

    # Bucle principal del algoritmo
    for t in range(T):
        E0 = 2 * np.random.rand(N) - 1
        J = 2 * (1 - np.random.rand(N))

        for i in range(N):
            r = np.random.rand()
            E = E0[i] * (1 - t / T)

            if abs(E) >= 1:
                # Fase de exploración
                Xrand = X[np.random.randint(0, N)]
                X[i] = Xrand - np.random.rand() * np.abs(Xrand - 2 * np.random.rand() * X[i])
            else:
                # Fase de explotación
                if r >= 0.5 and abs(E) >= 0.5:
                    # Soft besiege
                    X[i] = Xrabbit - E * np.abs(J[i] * Xrabbit - X[i])
                elif r >= 0.5 and abs(E) < 0.5:
                    # Hard besiege
                    X[i] = Xrabbit - E * np.abs(Xrabbit - X[i])
                elif r < 0.5 and abs(E) >= 0.5:
                    # Soft besiege with progressive rapid dives
                    X[i] = Xrabbit - E * np.abs(J[i] * Xrabbit - X[i]) + np.random.rand() * (Xrabbit - X[i])
                else:
                    # Hard besiege with progressive rapid dives
                    X[i] = Xrabbit - E * np.abs(Xrabbit - X[i]) + np.random.rand() * (Xrabbit - X[i])
            
            # Actualizar si la nueva solución es mejor
            if es_solucion_valida(X[i]):
                current_fitness = calcular_calidad(X[i]) - calcular_costo(X[i])
                if current_fitness > fitness[i]:
                    fitness[i] = current_fitness
                    if current_fitness > best_fitness:
                        best_fitness = current_fitness
                        Xrabbit = X[i].copy()
    
    return Xrabbit, best_fitness

# Ejecución del algoritmo
N = 50  # Tamaño de la población
T = 100  # Número máximo de iteraciones
max_cost = 10000  # Costo máximo permitido
best_location, best_value = hho(N, T)
print("Mejor ubicación:", best_location)
print("Mejor valor de aptitud:", best_value)


"""
# Ejemplo de código en Python usando PuLP para la formulación inicial
import pulp

# Crear el problema de LP para maximizar
model = pulp.LpProblem("Campaña_Publicitaria", pulp.LpMaximize)

# Variables de decisión
x = pulp.LpVariable.dicts("x", ['TV_tarde', 'TV_noche', 'Diarios', 'Revistas', 'Radio'], lowBound=0, cat='Integer')

# Función Objetivo
model += pulp.lpSum([q[i] * x[i] for i in x]), "Calidad_Total"

# Restricciones
model += pulp.lpSum([c['TV_tarde'] * x['TV_tarde'] + c['TV_noche'] * x['TV_noche']]) <= 3800, "Presupuesto_TV"
model += pulp.lpSum([c['Diarios'] * x['Diarios'] + c['Revistas'] * x['Revistas']]) <= 2800, "Presupuesto_Diarios_Revistas"
model += pulp.lpSum([c['Diarios'] * x['Diarios'] + c['Radio'] * x['Radio']]) <= 3500, "Presupuesto_Diarios_Radio"

# Resolviendo el modelo
model.solve()

# Imprimir solución óptima
for v in model.variables():
    print(v.name, "=", v.varValue)

# Implementación adicional del algoritmo HHO se haría tras validar este modelo


"""