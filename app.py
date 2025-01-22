from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Описание уравнений с формулами
equations = {
    "heat": {
        "title": "Уравнение теплопроводности",
        "formula": r"\frac{\partial u}{\partial t} = k \frac{\partial^2 u}{\partial x^2}",
        "description": "Уравнение теплопроводности описывает процесс передачи тепла в материале.",
        "parameters": {
            "t": "Время \\( t \\)",
            "k": "Коэффициент теплопроводности \\( k \\)",
            "x_min": "Нижняя граница \\( x_{min} \\)",
            "x_max": "Верхняя граница \\( x_{max} \\)"
        }
    },
    "wave": {
        "title": "Волновое уравнение",
        "formula": r"\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}",
        "description": "Волновое уравнение описывает распространение волн, например, звуковых или световых.",
        "parameters": {
            "t": "Время \\( t \\)",
            "c": "Скорость волны \\( c \\)",
            "x_min": "Нижняя граница \\( x_{min} \\)",
            "x_max": "Верхняя граница \\( x_{max} \\)"
        }
    },
    "advection": {
        "title": "Уравнение переноса",
        "formula": r"\frac{\partial u}{\partial t} + v \frac{\partial u}{\partial x} = 0",
        "description": "Уравнение переноса моделирует движение вещества или тепла в потоке.",
        "parameters": {
            "v": "Скорость потока \\( v \\)",
            "x_min": "Нижняя граница \\( x_{min} \\)",
            "x_max": "Верхняя граница \\( x_{max} \\)"
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
    plot_url = None
    params = request.form.to_dict() if request.method == 'POST' else {}

    if request.method == 'POST':
        try:
            # Получение начальных границ
            x_min = float(params.get("x_min", 0))
            x_max = float(params.get("x_max", 10))
            x = np.linspace(x_min, x_max, 100)
            fig, ax = plt.subplots()

            # Получение параметров для нескольких графиков
            t_values = [float(v.strip()) for v in params.get("t", "1").split(",")]
            k_values = [float(v.strip()) for v in params.get("k", "1").split(",")]
            c_values = [float(v.strip()) for v in params.get("c", "1").split(",")]
            v_values = [float(v.strip()) for v in params.get("v", "1").split(",")]

            # Уравнение теплопроводности
            if eq_name == "heat":
                num_points = 100
                dx = (x_max - x_min) / (num_points - 1)
                for k in k_values:
                    for t in t_values:
                        dt = dx**2 / (4 * k)  # Условие устойчивости
                        steps = int(t / dt)
                        u = np.zeros(num_points)
                        u[num_points // 2] = 1  # Начальное распределение
                        u_next = np.zeros_like(u)
                        for _ in range(steps):
                            u_next[1:-1] = u[1:-1] + k * dt / dx**2 * (u[:-2] - 2 * u[1:-1] + u[2:])
                            u = u_next.copy()
                        ax.plot(np.linspace(x_min, x_max, num_points), u, label=f"t={t}, k={k}")
                ax.set_title("Решение уравнения теплопроводности")

            # Волновое уравнение
            elif eq_name == "wave":
                num_points = 100
                dx = (x_max - x_min) / (num_points - 1)
                dt = 0.01
                steps = int(t_values[0] / dt)
                u_prev = np.zeros(num_points)
                u = np.zeros(num_points)
                u_next = np.zeros_like(u)
                u[num_points // 2] = 1  # Начальное распределение
                c = c_values[0]
                for _ in range(steps):
                    u_next[1:-1] = 2 * u[1:-1] - u_prev[1:-1] + c**2 * dt**2 / dx**2 * (u[:-2] - 2 * u[1:-1] + u[2:])
                    u_prev = u.copy()
                    u = u_next.copy()
                ax.plot(np.linspace(x_min, x_max, num_points), u, label=f"t={t_values[0]}, c={c}")
                ax.set_title("Решение волнового уравнения")

            # Уравнение переноса
            elif eq_name == "advection":
                num_points = 100
                dx = (x_max - x_min) / (num_points - 1)
                dt = dx / v_values[0]  # Условие Куранта
                steps = int(t_values[0] / dt)
                u = np.zeros(num_points)
                u[num_points // 2] = 1  # Начальное распределение
                u_next = np.zeros_like(u)
                for _ in range(steps):
                    u_next[1:] = u[1:] - v_values[0] * dt / dx * (u[1:] - u[:-1])
                    u = u_next.copy()
                ax.plot(np.linspace(x_min, x_max, num_points), u, label=f"t={t_values[0]}, v={v_values[0]}")
                ax.set_title("Решение уравнения переноса")

            ax.set_xlabel("x")
            ax.set_ylabel("u(x, t)")
            ax.legend()
            ax.grid(True)

            # Конвертация графика в Base64 для отображения на странице
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
            plt.close()

        except ValueError:
            plot_url = None

    return render_template('equation.html', eq_data=eq_data, plot_url=plot_url, params=params)

if __name__ == '__main__':
    app.run(debug=True)