"""
Punto de entrada del laboratorio de Copeland y Galai (1983).

Corre el flujo completo con un solo comando (python main.py):
    1. Optimiza el spread bid-ask (src/model.py) y muestra el resultado.
    2. Simula 10,000 trades y corre el análisis de Monte Carlo
       (src/simulation.py) para los tres regímenes de cotización.
    3. Genera y guarda las cuatro figuras del laboratorio (src/plots.py).
"""

from src import model
from src import simulation
from src import plots
import pytest


def main():
    print("=== 1. Optimización del spread (src/model.py) ===")
    resultado = model.optimizar_spread()
    print(f"Bid óptimo:            {resultado['bid']}")
    print(f"Ask óptimo:            {resultado['ask']}")
    print(f"Spread:                {resultado['spread']}")
    print(f"Utilidad esperada:     {resultado['utilidad_esperada']}")

    print("\n=== 1b. Análisis de sensibilidad respecto a πᵢ (src/model.py) ===")
    resultados_sensibilidad = model.analisis_sensibilidad()
    for r in resultados_sensibilidad:
        print(f"πᵢ={r['pi_i']}: spread numérico={r['spread']}, "
              f"spread teórico={r['spread_teorico']}")

    print("\n=== 2a. Simulación de 10,000 trades por régimen (src/simulation.py) ===")
    trades = simulation.simular_todos_los_regimenes(n_trades=10_000)
    print(trades.groupby("regimen")["pnl"].describe())

    print("\n=== 2b. Monte Carlo: 1,000 corridas de 1,000 trades por régimen ===")
    pnl_montecarlo = simulation.monte_carlo_pnl_por_regimen(n_corridas=1_000, n_trades=1_000)
    resumen_montecarlo = simulation.resumir_montecarlo(pnl_montecarlo)
    print(resumen_montecarlo.to_string(index=False))

    print("\n=== 3. Generando figuras (src/plots.py) ===")
    rutas = plots.generar_figuras(trades, pnl_montecarlo, resumen_montecarlo)
    ruta_sensibilidad = plots.graficar_sensibilidad(resultados_sensibilidad)
    rutas.append(ruta_sensibilidad)
    for ruta in rutas:
        print(f"Figura guardada: {ruta}")

    print("\n=== 4. Corriendo pruebas (tests/test_model.py) ===")
    codigo_salida = pytest.main(["tests/", "-v"])
    if codigo_salida == 0:
        print("Todas las pruebas pasaron correctamente.")
    else:
        print("Alguna prueba falló — revisa el detalle arriba.")


if __name__ == "__main__":
    main()

