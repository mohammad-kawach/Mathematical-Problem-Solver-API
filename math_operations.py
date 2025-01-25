import sympy as sp
import numpy as np
from graph_utils import generate_graph, generate_graph_system


def solve_math(user_input):
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
                    system_equations.append(sp.sympify(eq))

            solutions = sp.solve(system_equations)
            steps = generate_steps_system(system_equations, solutions)
            graph_url = generate_graph_system(system_equations)
            return solutions, steps, graph_url
        except Exception as e:
            raise ValueError("Invalid system of equations.")

    # Handle single expression
    try:
        parsed_input = sp.sympify(user_input)
        simplified = sp.simplify(parsed_input)
        graph_url = generate_graph(parsed_input)
        return simplified, None, graph_url
    except (sp.SympifyError, TypeError):
        raise ValueError("Invalid mathematical expression.")


def generate_steps_system(equations, solutions):
    steps = [f"Original system of equations:"]
    for eq in equations:
        steps.append(f"  {eq}")
    steps.append(f"Solutions: {solutions}")
    return steps
