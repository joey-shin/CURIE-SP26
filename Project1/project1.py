import sys

from atom import Atom, parse, parse_atom, print_atoms
from eval import Energy, Force, GD, GDLS, BFGS
from eval import GoldenSection


def write(string, fname):
    with open(fname, "a") as f:
        f.write(f"{string}\n")


def main(argv):
    if len(argv) != 3:
        raise SystemExit('Usage: python project1.py input.in output.out')

    input = argv[1]
    output = argv[2]

    try:
        config, atoms = parse(input)
    except Exception as e:
        print(f'Error parsing input: {e}', file=sys.stderr)
        raise SystemExit(1)
    
    if config.get('LINESEARCH') == 'GoldenSection':
        LS = GoldenSection

    if config['JOB'] == 'Energy':
        E = Energy(atoms)
        string = f"Energy = {E} kJ/mol"
        write(string, output)

    if config['JOB'] == 'Force':
        F = Force(atoms)
        string = f"Force: \n {F}"
        write(string, output)

    if config['JOB'] == 'GD':
        atoms_opt, E_opt, iterations = GD(atoms, 0.3, 1e-2, fname=output, verbose=config['VERBOSE'])
        string = f"""
------------------------
Total iterations: {iterations} 
Optimized structure: \n\n{print_atoms(atoms_opt)} 
Final Energy: {E_opt:.3e} kJ/mol
                """
        write(string, output)

    if config['JOB'] == 'GDLS':
        atoms_opt, E_opt, iterations = GDLS(atoms, 0.3, 1e-2, LS, fname=output, verbose=config['VERBOSE'])
        string = f"""
------------------------
Total iterations: {iterations} 
Optimized structure: \n\n{print_atoms(atoms_opt)} 
Final Energy: {E_opt:.3e} kJ/mol
                """
        write(string, output)

    if config['JOB'] == 'BFGS':
        string = f'Starting BFGS minimization with {config['LINESEARCH']} Line Search'
        write(string, output)
        atoms_opt, E_opt, iterations = BFGS(atoms, 0.3, 1e-2, fname=output, verbose=config['VERBOSE'])
        string = f"""
------------------------
Total iterations: {iterations} 
Optimized structure: \n\n{print_atoms(atoms_opt)} 
Final Energy: {E_opt:.3e} kJ/mol
                """
        write(string, output)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

