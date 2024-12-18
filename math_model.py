import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from math import exp, radians, pi

# Общие константы
G = 6.67e-11
earth_mass = 6e24
earth_radius = 6371000  # Радиус Земли
G_M_Earth = 6.674 * 5.9722 * 10 ** 13  # Гравитационная постоянная * масса Земли

# Общие параметры ракеты
S = 3.14 * 3.7 ** 2  # Площадь наибольшего поперечного сечения (миделево сечения), м^2
Cx = 0.5


# Время работы каждого из этапов
T1 = 124  # I ступень
T2 = 206  # II ступень
T3 = 238  # III ступень
T4 = 500  # Автономный полет
T5 = 365  # 1-ый запуск РБ "ДМ"
T6 = 3000  # Автономный полет
T7 = 365 # 2-ый запуск РБ "ДМ"
T8 = 20000  # Автономный полет

# Массы 
M1 = 450_510  # I ступень
M2 = 167_728  # II ступень
M3 = 50_747  # III ступень
M4_1 = 6.565 * 1000  # РБ "ДМ" (1-й этап)
M4_2 = 5.871 * 1000  # РБ "ДМ" (2-ой этап)



# Зависимость температуры от высоты 
def temperature(height):
    if height <= 11000:
        return 293 - 6.5 * height / 1000
    elif 11001 <= height <= 20000:
        return 221.65
    elif 20001 <= height <= 32000:
        return 216.65 + (height - 20000) / 1000
    elif 32001 <= height <= 40000:
        return temperature(32000) + 2.75 * (height - 32000) / 1000
    elif 40001 <= height <= 50000:
        return temperature(40000) + 2 * (height - 40000) / 1000
    elif 50001 <= height <= 60000:
        return temperature(50000) - 2.3 * (height - 50000) / 1000
    elif 60001 <= height <= 80000:
        return temperature(60000) - 2.45 * (height - 60000) / 1000
    elif 80001 <= height <= 100000:
        return temperature(80000) - 0.1 * (height - 80000) / 1000
    else:
        return temperature(100000) + 8.62 * (height - 100000) / 1000


# Зависимость давления от высоты
def pressure(height):
    return 101_135 * exp(-0.029 * g(height) * height / (8.31 * temperature(height)))


# Зависимость плотности воздуха от высоты
def density(height):
    return pressure(height) * 0.029 / (8.31 * temperature(height))

def g(height):
    return G * earth_mass / (earth_radius + height) ** 2

# Сила сопротивления воздуха
def get_resistance(r, phi, r_dot, phi_dot):
    return Cx * density(r - earth_radius) * (r_dot ** 2 + (r_dot * phi_dot) ** 2) * S / 2


# Следующие 3 метода описывают полёт ракеты - разгон с учётом атмосферы, разгон без учёта атмосферы, и автономный полёт
def first_stage(initial_conditions, T, F, sigma, M, k, beta_start, beta_end):
    beta_incr = radians((beta_end - beta_start) / T)
    beta_start = radians(beta_start)
    def right_part(t, y):
        y1, y2, y3, y4 = y 
        return [
            y2, 
            y1 * (y4 ** 2) - G_M_Earth / (y1 ** 2) + (np.cos(beta_start + beta_incr * t) / (M - k * t)) * (
                    (F + sigma * t) - get_resistance(y1, y3, y2, y4)),
            y4, 
            (4000000 * np.sin(beta_start + beta_incr * t) * ((F + sigma * t) - get_resistance(y1, y3, y2, y4)) / (
                    M - k * t) - 2 * y2 * y1 * y4) / (y1 ** 2)
        ]
    t = np.array([i for i in range(0, T, 1)])
    solv = solve_ivp(right_part, [0, T], initial_conditions, method='RK45', dense_output=True)
    rez = solv.sol(t) # набор массивов [t, y]
    return rez

def second_stage(initial_conditions, T, F, M, k, beta_start, beta_end):
    beta_incr = radians((beta_end - beta_start) / T)
    beta_start = radians(beta_start)
    def right_part(t, y):
        y1, y2, y3, y4 = y
        return [
            y2,
            y1 * (y4 ** 2) - G_M_Earth / (y1 ** 2) + (np.cos(beta_start + beta_incr * t) / (M - k * t)) * F,
            y4,
            (4000000 * np.sin(beta_start + beta_incr * t) * F / (M - k * t) - 2 * y2 * y1 * y4) / (y1 ** 2)
        ]
    t = np.array([i for i in range(0, T, 1)])
    solv = solve_ivp(right_part, [0, T], initial_conditions, method='RK45', dense_output=True)
    rez = solv.sol(t)
    return rez

def autonomous_flight(initial_conditions, T):
    def right_part(t, y):
        y1, y2, y3, y4 = y
        return [
            y2,
            y1 * (y4 ** 2) - G_M_Earth / (y1 ** 2),
            y4,
            -2 * ((y4 * y2) / y1)]
    t = np.array([i for i in range(0, T, 1)])
    solv = solve_ivp(right_part, [0, T], initial_conditions, method='RK45', dense_output=True)
    rez = solv.sol(t)
    return rez

# Подсчет траектории на всех этапах
def get_trajectory(start_pos):
    trajectory = []
    trajectory.append(
        first_stage(start_pos, T1, 10026 * 1000, 7983.5, M1 + M2 + M3 + M4_1, 3622, 0, 60))
    trajectory.append(
        first_stage(trajectory[-1][:, -1], T2, 2400 * 1000, 0, M2 + M3 + M4_1, 731.63, 60, 60))
    trajectory.append(second_stage(trajectory[-1][:, -1], T3, 583 * 1000, M3 + M4_1, 180, 60, 60))
    trajectory.append(autonomous_flight(trajectory[-1][:, -1], T4))
    trajectory.append(second_stage(trajectory[-1][:, -1], T5, 150 * 1000, M4_1, 2.57, 60, 80))
    trajectory.append(autonomous_flight(trajectory[-1][:, -1], T6))
    trajectory.append(second_stage(trajectory[-1][:, -1], T7, 32.2 * 1000, M4_2, 2.57, 90, 90))
    trajectory.append(autonomous_flight(trajectory[-1][:, -1], T8))
    return trajectory

def join_flight_stages(trajectory):
    t = np.array([i for i in range(0, T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8, 1)])
    r = np.concatenate([stage[0, :] for stage in trajectory])
    r_dot = np.concatenate([stage[1, :] for stage in trajectory])
    phi = np.concatenate([stage[2, :] for stage in trajectory])
    phi_dot = np.concatenate([stage[3, :] for stage in trajectory])
    return t, np.array([r, r_dot, phi, phi_dot])

# График высоты
def draw_height(axis, trajectory):
    t, stages = join_flight_stages(trajectory)
    h = stages[0, :] - earth_radius
    axis.plot(t[:28000], h[:28000], color='black')
    axis.grid()

# График скорости
def draw_speed(axis, trajectory):
    t, stages = join_flight_stages(trajectory)
    v = np.sqrt(stages[1, :] ** 2 + (stages[0, :] * stages[3, :]) ** 2)
    axis.plot(t[:28000], v[:28000], color='black')
    axis.grid()


def show_flight_parameter_plots(trajectory):
    fig, axs = plt.subplots(nrows=2, figsize=(8, 8))
    axs[0].set_xlabel('t, с')
    axs[0].set_ylabel('h, м')
    axs[1].set_xlabel('t, с')
    axs[1].set_ylabel('V, м/с')

    draw_height(axs[0], trajectory)
    draw_speed(axs[1], trajectory)
    plt.show()


trajectory = get_trajectory([earth_radius, 0, 0, 0])
show_flight_parameter_plots(trajectory)