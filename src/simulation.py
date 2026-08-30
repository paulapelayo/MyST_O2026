"""
Simulador de trades para el modelo de Copeland y Galai (1983).

Simula la llegada de contrapartes al market maker bajo tres regímenes de
cotización (óptimo, estrecho y amplio) y corre un análisis de Monte Carlo
sobre el P&L acumulado del market maker en cada régimen.
"""

import numpy as np
import pandas as pd
from scipy.stats import erlang

from src import model

# Semilla fija para que la simulación sea reproducible (documentar en README).
np.random.seed(42)


# ---------------------------------------------------------------------------
# Regímenes de cotización a comparar
# ---------------------------------------------------------------------------

_OPTIMO = model.optimizar_spread()

REGIMENES = {
    "optimo": (_OPTIMO["bid"], _OPTIMO["ask"]),
    "estrecho": (19.75, 20.05),
    "amplio": (18.40, 21.40),
}


def _generar_pnl(bid, ask, n):
    """
    Simula n llegadas de contrapartes contra un Bid/Ask fijo.

    Para cada llegada se decide primero si la contraparte es un trader
    informado (con probabilidad PI_I) o de liquidez (con probabilidad
    PI_L). Un trader informado solo opera si le conviene: compra al Ask
    si el valor verdadero P (simulado con la Erlang de src/model.py) es
    mayor al Ask, o vende al Bid si P es menor al Bid; si P cae dentro
    del spread, no tiene ventaja de información y no opera. Un trader de
    liquidez decide comprar o vender al azar (50/50) y su operación se
    ejecuta con la probabilidad pi_LB/pi_LS de src/model.py, que baja
    entre más lejos esté el precio pactado del precio de referencia S0.

    Regresa, para las n llegadas: el tipo de contraparte, el lado
    ejecutado ('compra', 'venta' o 'sin_trade' si no se ejecutó), si se
    ejecutó o no, el P&L del trade (desde el punto de vista del market
    maker) y el cambio en su inventario.
    """
    informado = np.random.rand(n) < model.PI_I
    tipo = np.where(informado, "informado", "liquidez")

    pnl = np.zeros(n)
    lado = np.full(n, "sin_trade", dtype=object)
    inventario_delta = np.zeros(n, dtype=int)

    # --- Traders informados: operan solo cuando les conviene ---
    idx_inf = np.where(informado)[0]
    if idx_inf.size:
        P = erlang.rvs(model.K, scale=1 / model.LAMBDA, size=idx_inf.size)
        compra = P > ask   # el valor real supera el Ask: al informado le conviene comprar
        venta = P < bid    # el valor real es menor al Bid: al informado le conviene vender

        pnl[idx_inf[compra]] = ask - P[compra]   # pérdida del market maker
        pnl[idx_inf[venta]] = P[venta] - bid     # pérdida del market maker

        lado[idx_inf[compra]] = "compra"
        lado[idx_inf[venta]] = "venta"

        inventario_delta[idx_inf[compra]] = -1   # el market maker vendió
        inventario_delta[idx_inf[venta]] = 1     # el market maker compró

    # --- Traders de liquidez: lado al azar, ejecución probabilística ---
    idx_liq = np.where(~informado)[0]
    if idx_liq.size:
        quiere_comprar = np.random.rand(idx_liq.size) < 0.5

        idx_compra = idx_liq[quiere_comprar]
        if idx_compra.size:
            prob_ejecucion = model.pi_LB(ask - model.S0)
            ejecuta = np.random.rand(idx_compra.size) < prob_ejecucion
            ejecutados = idx_compra[ejecuta]
            pnl[ejecutados] = ask - model.S0     # ganancia del market maker
            lado[ejecutados] = "compra"
            inventario_delta[ejecutados] = -1

        idx_venta = idx_liq[~quiere_comprar]
        if idx_venta.size:
            prob_ejecucion = model.pi_LS(model.S0 - bid)
            ejecuta = np.random.rand(idx_venta.size) < prob_ejecucion
            ejecutados = idx_venta[ejecuta]
            pnl[ejecutados] = model.S0 - bid     # ganancia del market maker
            lado[ejecutados] = "venta"
            inventario_delta[ejecutados] = 1

    ejecutado = lado != "sin_trade"
    return tipo, lado, ejecutado, pnl, inventario_delta


def simular_trades(bid, ask, n_trades=10_000):
    """
    Simula n_trades llegadas de contrapartes contra un Bid/Ask fijo y las
    regresa en un DataFrame listo para graficar, con una fila por llegada
    (incluye las que no se ejecutaron, marcadas en la columna 'ejecutado').
    """
    tipo, lado, ejecutado, pnl, inventario_delta = _generar_pnl(bid, ask, n_trades)
    return pd.DataFrame({
        "tipo_trader": tipo,
        "lado": lado,
        "ejecutado": ejecutado,
        "pnl": pnl,
        "inventario_delta": inventario_delta,
    })


def simular_todos_los_regimenes(n_trades=10_000):
    """
    Corre simular_trades para los tres regímenes de REGIMENES (óptimo,
    estrecho, amplio) y regresa un solo DataFrame combinado, con una
    columna 'regimen' para poder comparar o filtrar al graficar.
    """
    tablas = []
    for nombre, (bid, ask) in REGIMENES.items():
        tabla = simular_trades(bid, ask, n_trades)
        tabla["regimen"] = nombre
        tablas.append(tabla)
    return pd.concat(tablas, ignore_index=True)


def monte_carlo_regimen(bid, ask, n_corridas=1_000, n_trades=1_000):
    """
    Corre n_corridas simulaciones independientes de n_trades llegadas cada
    una contra un Bid/Ask fijo, y regresa el P&L final (acumulado) del
    market maker en cada corrida.
    """
    _, _, _, pnl, _ = _generar_pnl(bid, ask, n_corridas * n_trades)
    pnl_por_corrida = pnl.reshape(n_corridas, n_trades)
    return pnl_por_corrida.sum(axis=1)


def monte_carlo_regimenes(n_corridas=1_000, n_trades=1_000):
    """
    Resume el análisis de Monte Carlo para los tres regímenes de
    REGIMENES: P&L promedio, desviación estándar del P&L y probabilidad
    de pérdida (proporción de corridas cuyo P&L final resultó negativo).
    """
    filas = []
    for nombre, (bid, ask) in REGIMENES.items():
        pnl_final = monte_carlo_regimen(bid, ask, n_corridas, n_trades)
        filas.append({
            "regimen": nombre,
            "pnl_promedio": round(float(pnl_final.mean()), 4),
            "pnl_std": round(float(pnl_final.std()), 4),
            "prob_perdida": round(float((pnl_final < 0).mean()), 4),
        })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    print(REGIMENES)
    print(simular_todos_los_regimenes().groupby("regimen")["pnl"].describe())
    print(monte_carlo_regimenes())
