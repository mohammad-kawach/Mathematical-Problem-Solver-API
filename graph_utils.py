import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import sympy as sp


def generate_graph(expression):
    try:
        x = sp.symbols('x')
        f = sp.lambdify(x, expression, 'numpy')

        x_vals = np.linspace(-10, 10, 400)
        y_vals = f(x_vals)

        mask = np.isfinite(y_vals) & np.isreal(y_vals)

        plt.figure(figsize=(6, 4))
        plt.plot(x_vals[mask], y_vals[mask], label=str(expression))
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.title(f"Graph of {expression}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"
        plt.close()
        return graph_url
    except Exception as e:
        return None


def generate_graph_system(equations):
    try:
        x, y = sp.symbols('x y')
        plt.figure(figsize=(8, 6))

        x_vals = np.linspace(-10, 10, 400)

        for eq in equations:
            try:
                y_expr = sp.solve(eq, y)[0]
                f = sp.lambdify(x, y_expr, 'numpy')
                y_vals = f(x_vals)

                mask = np.isfinite(y_vals) & np.isreal(y_vals)
                plt.plot(x_vals[mask], y_vals[mask], label=str(eq))
            except:
                continue

        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.title("System of Equations")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"
        plt.close()
        return graph_url
    except Exception as e:
        return None
