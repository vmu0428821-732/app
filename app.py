from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Добавлено для корректной работы в Flask
from matplotlib.animation import FuncAnimation
import uuid
import os
import time
import atexit
import logging
import traceback

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Создаем папку static если ее нет
static_dir = os.path.join(app.root_path, 'static')
print(f"Путь к папке static: {static_dir}")
try:
    os.makedirs(static_dir, exist_ok=True)
    print(f"Папка static создана или уже существует")
    # Проверяем права на запись
    test_file = os.path.join(static_dir, 'test.txt')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    print(f"Права на запись в папку static есть")
except Exception as e:
    print(f"Ошибка при работе с папкой static: {e}")

# Очистка старых файлов при завершении
@atexit.register
def cleanup():
    for file in os.listdir(static_dir):
        if file.startswith('animation_') and file.endswith('.gif'):
            os.remove(os.path.join(static_dir, file))

# Описание уравнений с формулами
equations = {
    "heat": {
        "title": "Уравнение теплопроводности",
        "formula": r"\frac{\partial u}{\partial t} = k \frac{\partial^2 u}{\partial x^2}",
        "description": "Уравнение теплопроводности описывает процесс передачи тепла в материале.",
        "parameters": {
            "x_start": "Начало отрезка \\( x_{start} \\)",
            "x_end": "Конец отрезка \\( x_{end} \\)",
            "N": "Количество разбиений \\( N \\)",
            "t_end": "Конечный момент времени \\( t_{end} \\)",
            "k": "Коэффициент теплопроводности \\( k \\)",
            "cfl": "Коэффициент устойчивости (CFL) \\( \\alpha \\)"
        }
    },
    "wave": {
        "title": "Волновое уравнение",
        "formula": r"\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}",
        "description": "Волновое уравнение описывает распространение волн, например, звуковых или световых.",
        "parameters": {
            "x_start": "Начало отрезка \\( x_{start} \\)",
            "x_end": "Конец отрезка \\( x_{end} \\)",
            "N": "Количество разбиений \\( N \\)",
            "t_end": "Конечный момент времени \\( t_{end} \\)",
            "c": "Скорость волны \\( c \\)",
            "cfl": "Коэффициент устойчивости (CFL) \\( \\alpha \\)"
        }
    },
    "advection": {
        "title": "Уравнение переноса",
        "formula": r"\frac{\partial u}{\partial t} + v \frac{\partial u}{\partial x} = 0",
        "description": "Уравнение переноса моделирует движение вещества или тепла в потоке.",
        "parameters": {
            "x_start": "Начало отрезка \\( x_{start} \\)",
            "x_end": "Конец отрезка \\( x_{end} \\)",
            "N": "Количество разбиений \\( N \\)",
            "t_end": "Конечный момент времени \\( t_{end} \\)",
            "v": "Скорость потока \\( v \\)",
            "cfl": "Коэффициент устойчивости (CFL) \\( \\alpha \\)"
        }
    }
}

# Главная страница
@app.route('/')
def index():
    return render_template('index.html', equations=equations)

# Страница ввода данных
@app.route('/equation/<eq_name>', methods=['GET', 'POST'])
def equation_page(eq_name):
    if eq_name not in equations:
        return redirect(url_for('index'))
    
    eq_data = equations[eq_name]
    params = request.form.to_dict() if request.method == 'POST' else {}
    animation_url = None
    t_max = None

    if request.method == 'POST':
        print("Получен POST запрос")
        try:
            # Общие параметры
            print("Получаемые параметры:", params)
            x_start = float(params.get("x_start", 0))
            x_end = float(params.get("x_end", 10))
            N = int(params.get("N", 100))
            t_end = float(params.get("t_end", 1.0))
            
            # Чтение граничных условий от пользователя
            left_boundary_value = float(params.get("left_boundary_value", 0))
            right_boundary_value = float(params.get("right_boundary_value", 0))
            
            if N < 10:
                raise ValueError("Количество разбиений должно быть не менее 10")
            
            # Вычисляем шаг по пространству
            delta_x = (x_end - x_start) / N
            
            # Инициализация сетки
            x = np.linspace(x_start, x_end, N+1)
            
            # Массив для хранения решения
            u_history = []
            t_history = []
            
            if eq_name == 'heat':
                k = float(params.get("k", 0.5))
                cfl = float(params.get("cfl", 0.9))
                if k <= 0 or k > 1:
                    raise ValueError("Коэффициент теплопроводности должен быть в диапазоне 0 < k ≤ 1")
                if cfl <= 0 or cfl > 1:
                    raise ValueError("Коэффициент устойчивости (CFL) должен быть в диапазоне 0 < CFL ≤ 1")
                
                # Начальное условие - синусоидальная функция
                u = np.zeros(N+1)
                for i in range(N+1):
                    u[i] = 0.5 * (1 + np.sin(2 * np.pi * (x[i] - x_start) / (x_end - x_start)))
                
                # Численное решение
                t = 0
                while t < t_end:
                    # Расчёт временного шага с учетом CFL
                    delta_t = cfl * (delta_x ** 2) / (2 * k)
                    
                    # Если следующий шаг выведет за t_end, уменьшаем его
                    if t + delta_t > t_end:
                        delta_t = t_end - t
                    
                    # Вычисление нового решения
                    u_new = u.copy()
                    for i in range(1, N):
                        u_new[i] = u[i] + (k * delta_t / delta_x**2) * (u[i+1] - 2*u[i] + u[i-1])
                    
                    # Граничные условия от пользователя
                    u_new[0] = left_boundary_value
                    u_new[N] = right_boundary_value
                    
                    # Обновление решения
                    u = u_new
                    t += delta_t
                    
                    # Сохранение истории
                    u_history.append(u.copy())
                    t_history.append(t)
            
            elif eq_name == 'wave':
                c = float(params.get("c", 1.0))
                cfl = float(params.get("cfl", 0.9))
                if c <= 0:
                    raise ValueError("Скорость волны должна быть положительной")
                if cfl <= 0 or cfl > 1:
                    raise ValueError("Коэффициент устойчивости (CFL) должен быть в диапазоне 0 < CFL ≤ 1")
                
                # Начальное условие согласованное с граничными условиями
                u = np.zeros(N+1)
                for i in range(N+1):
                    xi = x[i]
                    L = x_end - x_start
                    if left_boundary_value == 0 and right_boundary_value == 0:
                        # Если граничные условия нулевые, используем только синусоидальную составляющую
                        u[i] = np.sin(np.pi * (xi - x_start) / L)
                    else:
                        # Иначе используем линейную интерполяцию и синусоидальную составляющую
#                        u[i] = left_boundary_value + (right_boundary_value - left_boundary_value) * (xi - x_start) / L \
#                            + np.sin(np.pi * (xi - x_start) / L) * 0.8 * (right_boundary_value - left_boundary_value)
                        u[i] = np.sin(np.pi * (xi - x_start) / L) + \
                            (right_boundary_value - left_boundary_value) * (xi - x_start) / (x_end - x_start) + left_boundary_value
                u[0] = left_boundary_value
                u[N] = right_boundary_value
                u_prev = u.copy()
                
                # Численное решение
                t = 0
                while t < t_end:
                    # Расчёт временного шага с учетом CFL
                    delta_t = cfl * delta_x / c
                    
                    # Если следующий шаг выведет за t_end, уменьшаем его
                    if t + delta_t > t_end:
                        delta_t = t_end - t
                    
                    # Вычисление нового решения
                    u_new = np.zeros(N+1)
                    for i in range(1, N):
                        u_new[i] = 2*u[i] - u_prev[i] + (c**2 * delta_t**2 / delta_x**2) * (u[i+1] - 2*u[i] + u[i-1])
                    
                    # Граничные условия от пользователя
                    u_new[0] = left_boundary_value
                    u_new[N] = right_boundary_value
                    
                    # Обновление решения
                    u_prev = u.copy()
                    u = u_new
                    t += delta_t
                    
                    # Сохранение истории
                    u_history.append(u.copy())
                    t_history.append(t)
            
            elif eq_name == 'advection':
                v = float(params.get("v", 1.0))
                cfl = float(params.get("cfl", 0.9))
                if v <= 0:
                    raise ValueError("Скорость потока должна быть положительной")
                if cfl <= 0 or cfl > 1:
                    raise ValueError("Коэффициент устойчивости (CFL) должен быть в диапазоне 0 < CFL ≤ 1")
                
                # Начальное условие - прямоугольный импульс
                x1 = x_start + (x_end - x_start) / 4  # Начало импульса
                x2 = x_start + (x_end - x_start) / 2  # Конец импульса
                u = np.zeros(N+1)
                for i in range(N+1):
                    if x1 <= x[i] <= x2:
                        u[i] = 1.0
                
                # Численное решение с использованием модифицированной схемы Лакса-Вендроффа
                t = 0
                # Коэффициент искусственной вязкости
                epsilon = 0.01
                
                while t < t_end:
                    # Расчёт временного шага с учетом CFL
                    delta_t = cfl * delta_x / abs(v)
                    
                    # Если следующий шаг выведет за t_end, уменьшаем его
                    if t + delta_t > t_end:
                        delta_t = t_end - t
                    
                    # Вычисление нового решения по схеме Лакса-Вендроффа с искусственной вязкостью
                    u_new = np.zeros(N+1)
                    
                    # Периодические граничные условия
                    u[0] = u[N-1]  # Левая граница равна предпоследней точке
                    u[N] = u[1]    # Правая граница равна второй точке
                    
                    for i in range(1, N):
                        # Схема Лакса-Вендроффа
                        lw_term = u[i] - (v * delta_t / (2 * delta_x)) * (u[i+1] - u[i-1]) + \
                                 (v**2 * delta_t**2 / (2 * delta_x**2)) * (u[i+1] - 2*u[i] + u[i-1])
                        
                        # Добавляем искусственную вязкость
                        viscosity = epsilon * (u[i+1] - 2*u[i] + u[i-1])
                        
                        u_new[i] = lw_term + viscosity
                    
                    # Периодические граничные условия для нового решения
                    u_new[0] = u_new[N-1]
                    u_new[N] = u_new[1]
                    
                    # Применяем фильтр для сглаживания высокочастотных осцилляций
                    u_filtered = np.zeros(N+1)
                    for i in range(1, N):
                        u_filtered[i] = 0.25 * u_new[i-1] + 0.5 * u_new[i] + 0.25 * u_new[i+1]
                    
                    # Периодические граничные условия для фильтрованного решения
                    u_filtered[0] = u_filtered[N-1]
                    u_filtered[N] = u_filtered[1]
                    
                    # Обновление решения
                    u = u_filtered.copy()
                    t += delta_t
                    
                    # Сохранение истории
                    u_history.append(u.copy())
                    t_history.append(t)
            
            # Визуализация
            fig, ax = plt.subplots(figsize=(12, 6))
            
            def animate(frame):
                ax.clear()
                ax.plot(x, u_history[frame], 'b-', linewidth=2)
                ax.set_title(f"Распределение в момент времени t = {t_history[frame]:.3f}")
                ax.set_xlabel("Пространственная координата x")
                ax.set_ylabel("Величина u(x,t)")
                if eq_name == 'wave':
                    ax.set_ylim(-1.0, 1.0)  # Устанавливаем пределы по y для волнового уравнения
                else:
                    ax.set_ylim(-0.2, 1.2)  # Для остальных уравнений оставляем прежние пределы
                ax.grid(True)
                return ax.get_lines()
            
            print("Создаем анимацию...")
            
            # Создаем уникальный идентификатор для этой анимации
            animation_id = str(uuid.uuid4())
            animation_dir = os.path.join(static_dir, f'animation_{animation_id}')
            os.makedirs(animation_dir, exist_ok=True)
            print(f"Создана директория для анимации: {animation_dir}")
            
            # После расчёта:
            # Вместо генерации PNG формируем frame_data для передачи в шаблон
            frame_data = []
            for i in range(len(u_history)):
                frame_data.append({
                    'x': x.tolist(),
                    'y': u_history[i].tolist(),
                    't': t_history[i],
                    'title': f"Распределение в момент времени t = {t_history[i]:.3f}"
                })
            # Передаем информацию об анимации в шаблон
            animation_data = {
                'frame_data': frame_data,
                'frame_count': len(frame_data),
                'interval': 50  # интервал между кадрами в миллисекундах
            }
            return render_template('equation.html',
                                 eq_data=eq_data,
                                 eq_name=eq_name,
                                 animation_data=animation_data,
                                 params=params,
                                 t_max=t_max)

        except ValueError as ve:
            print(f"Ошибка валидации: {str(ve)}")
            error_fields = []
            if "x_min" in str(ve):
                error_fields = ['x_min', 'x_max']
            elif "Коэффициент теплопроводности" in str(ve) or "условие устойчивости" in str(ve).lower():
                error_fields = ['k']
            elif "Время должно быть" in str(ve):
                error_fields = ['t']
            
            return render_template('equation.html',
                                 eq_data=eq_data,
                                 eq_name=eq_name,
                                 error_message=str(ve),
                                 error_type="validation",
                                 error_fields=error_fields,
                                 params=params)
        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")
            print(f"Тип ошибки: {type(e)}")
            print(f"Полный стек ошибки: {traceback.format_exc()}")
            return render_template('equation.html',
                                 eq_data=eq_data,
                                 eq_name=eq_name,
                                 error_message=f"Произошла ошибка: {str(e)}",
                                 params=params)

    return render_template('equation.html',
                         eq_data=eq_data,
                         eq_name=eq_name,
                         plot_url=animation_url,
                         params=params,
                         t_max=t_max)

if __name__ == '__main__':
    app.run(debug=True)
