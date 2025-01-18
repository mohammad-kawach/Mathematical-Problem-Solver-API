from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from flask_cors import CORS
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


app = Flask(__name__)
CORS(app)
load_dotenv()

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    print(f"Received input: {user_input}")

    try:
        result, steps, graph_url = solve_math(user_input)
        response = {"response": f"The solution is: {result}"}
        if steps:
            response["steps"] = steps
        if graph_url:
            response["graph"] = graph_url
        return jsonify(response)
    except ValueError as e:
        return jsonify({"response": str(e)})
    except Exception as e:
        return jsonify({"response": "Sorry, I couldn't solve that. Please make sure your input is a valid mathematical problem."})

def solve_math(user_input):
    print(f"Raw user input: {user_input}")

    # Split multiple equations
    equations = [eq.strip() for eq in user_input.split(',')]

    # Replace '^' with '**' for proper exponentiation
    equations = [eq.replace('^', '**') for eq in equations]

    # Add explicit multiplication for terms like 2x
    import re
    equations = [re.sub(r'(\d)([a-zA-Z])', r'\1*\2', eq) for eq in equations]

    # Define symbols
    x, y, z = sp.symbols('x y z')

    if any('=' in eq for eq in equations):
        try:
            system_equations = []
            for eq in equations:
                if '=' in eq:
                    lhs, rhs = eq.split('=')
                    lhs = sp.sympify(lhs.strip())
                    rhs = sp.sympify(rhs.strip())
                    system_equations.append(sp.Eq(lhs, rhs))
                else:
                    # Handle expressions without equals sign
                    system_equations.append(sp.sympify(eq))

            print(f"System equations: {system_equations}")

            # Solve the system of equations
            solutions = sp.solve(system_equations)
            steps = generate_steps_system(system_equations, solutions)
            graph_url = generate_graph_system(system_equations)
            return solutions, steps, graph_url
        except Exception as e:
            print(f"Error solving system: {e}")
            raise ValueError("Invalid system of equations.")

    # Handle single expression
    try:
        parsed_input = sp.sympify(user_input)
        print(f"Parsed input: {parsed_input}")
        simplified = sp.simplify(parsed_input)
        graph_url = generate_graph(parsed_input)
        return simplified, None, graph_url
    except (sp.SympifyError, TypeError) as e:
        print(f"SymPy parsing error: {e}")
        raise ValueError("Invalid mathematical expression.")

def generate_steps_system(equations, solutions):
    steps = []
    steps.append(f"Original system of equations:")
    for eq in equations:
        steps.append(f"  {eq}")
    steps.append(f"Solutions: {solutions}")
    return steps

def generate_graph(expression):
    """
    Generate a graph for a single expression.
    """
    try:
        # Create a plot
        x = sp.symbols('x')
        f = sp.lambdify(x, expression, 'numpy')

        # Generate x and y values
        x_vals = np.linspace(-10, 10, 400)
        y_vals = f(x_vals)

        # Filter out complex values and infinities
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

        # Set reasonable axis limits
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"
        plt.close()
        return graph_url
    except Exception as e:
        print(f"Error generating graph: {e}")
        return None

def generate_graph_system(equations):
    try:
        # Check if we have equations in two variables (x and y)
        x, y = sp.symbols('x y')

        # Create a plot
        plt.figure(figsize=(8, 6))

        # Generate points for each equation
        x_vals = np.linspace(-10, 10, 400)

        for eq in equations:
            try:
                # Solve equation for y in terms of x
                y_expr = sp.solve(eq, y)[0]
                f = sp.lambdify(x, y_expr, 'numpy')
                y_vals = f(x_vals)

                # Filter out complex values and infinities
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

        # Set reasonable axis limits
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"
        plt.close()
        return graph_url
    except Exception as e:
        print(f"Error generating graph: {e}")
        return None

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)