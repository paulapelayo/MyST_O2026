"""
Generación de figuras para el laboratorio de Copeland y Galai (1983).

Recibe los resultados ya calculados por src/model.py y src/simulation.py
(no vuelve a simular nada) y produce cuatro figuras:
    1. Densidad Erlang del valor verdadero del activo, con el Bid/Ask de
       cada régimen marcado encima.
    2. Distribución del P&L por trade (simulación de 10,000 trades) para
       los tres regímenes.
    3. Distribución del P&L final de Monte Carlo (1,000 corridas de 1,000
       trades) para los tres regímenes.
    4. Resumen comparativo: P&L promedio ± desviación estándar y
       probabilidad de pérdida por régimen.
"""

"""
Generación de figuras para el laboratorio de Copeland y Galai (1983).
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import erlang

try:
    from src import model
    from src import simulation
except ImportError:
    import model
    import simulation

_RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIRECTORIO_OUTPUT = os.path.join(_RAIZ_PROYECTO, "output")

REGIMEN_ORDEN = ["optimo", "estrecho", "amplio"]
REGIMEN_ETIQUETAS = {"optimo": "Óptimo", "estrecho": "Estrecho", "amplio": "Amplio"}
REGIMEN_COLORES = {"optimo": "#2ca02c", "estrecho": "#d62728", "amplio": "#1f77b4"}


def graficar_densidad_y_spreads(guardar_en=None):
    if guardar_en is None:
        guardar_en = os.path.join(_DIRECTORIO_OUTPUT, "01_densidad_y_spreads.png")

    lo = erlang.ppf(0.001, model.K, scale=1 / model.LAMBDA)
    hi = erlang.ppf(0.999, model.K, scale=1 / model.LAMBDA)
    P = np.linspace(lo, hi, 500)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(P, model.f(P), color="black", label="Densidad f(P)")
    ax.axvline(model.S0, color="gray", linestyle=":", label="S0 (referencia)")

    for nombre in REGIMEN_ORDEN:
        bid, ask = simulation.REGIMENES[nombre]
        color = REGIMEN_COLORES[nombre]
        ax.axvline(bid, color=color, linestyle="--", alpha=0.8)
        ax.axvline(ask, color=color, linestyle="--", alpha=0.8,
                    label=f"{REGIMEN_ETIQUETAS[nombre]} (Bid={bid}, Ask={ask})")

    ax.set_xlabel("Precio")
    ax.set_ylabel("Densidad")
    ax.set_title("Valor verdadero del activo y spreads Bid-Ask por régimen")
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(guardar_en), exist_ok=True)
    fig.savefig(guardar_en, dpi=150)
    plt.close(fig)
    return guardar_en


def graficar_pnl_por_trade(trades_df, guardar_en=None):
    if guardar_en is None:
        guardar_en = os.path.join(_DIRECTORIO_OUTPUT, "02_pnl_por_trade.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, nombre in zip(axes, REGIMEN_ORDEN):
        trades_regimen = trades_df.loc[trades_df["regimen"] == nombre]
        pnl_ejecutados = trades_regimen.loc[trades_regimen["ejecutado"], "pnl"]
        pct_no_ejecutados = 100 * (1 - trades_regimen["ejecutado"].mean())

        ax.hist(pnl_ejecutados, bins=60, color=REGIMEN_COLORES[nombre])
        ax.set_title(REGIMEN_ETIQUETAS[nombre])
        ax.set_xlabel("P&L por trade (solo ejecutados)")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.text(
            0.02, 0.95, f"No ejecutados: {pct_no_ejecutados:.1f}%",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    axes[0].set_ylabel("Número de trades")
    fig.suptitle("Distribución del P&L por trade (10,000 trades simulados)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(guardar_en), exist_ok=True)
    fig.savefig(guardar_en, dpi=150)
    plt.close(fig)
    return guardar_en


def graficar_montecarlo_pnl_final(pnl_por_regimen, guardar_en=None):
    if guardar_en is None:
        guardar_en = os.path.join(_DIRECTORIO_OUTPUT, "03_montecarlo_pnl_final.png")

    fig, ax = plt.subplots(figsize=(9, 5))

    for nombre in REGIMEN_ORDEN:
        ax.hist(
            pnl_por_regimen[nombre], bins=40, alpha=0.6,
            color=REGIMEN_COLORES[nombre], label=REGIMEN_ETIQUETAS[nombre],
        )

    ax.axvline(0, color="black", linewidth=0.8, label="Punto de equilibrio")
    ax.set_xlabel("P&L final por corrida (1,000 trades)")
    ax.set_ylabel("Número de corridas")
    ax.set_title("Monte Carlo: P&L final por régimen (1,000 corridas)")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(guardar_en), exist_ok=True)
    fig.savefig(guardar_en, dpi=150)
    plt.close(fig)
    return guardar_en


def graficar_resumen_regimenes(resumen_df, guardar_en=None):
    if guardar_en is None:
        guardar_en = os.path.join(_DIRECTORIO_OUTPUT, "04_resumen_regimenes.png")

    resumen_df = resumen_df.set_index("regimen").loc[REGIMEN_ORDEN]
    etiquetas = [REGIMEN_ETIQUETAS[nombre] for nombre in resumen_df.index]
    colores = [REGIMEN_COLORES[nombre] for nombre in resumen_df.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.bar(etiquetas, resumen_df["pnl_promedio"], yerr=resumen_df["pnl_std"],
            color=colores, capsize=6)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("P&L promedio (± 1 desv. est.)")
    ax1.set_title("P&L final promedio por régimen")

    ax2.bar(etiquetas, resumen_df["prob_perdida"], color=colores)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Probabilidad de pérdida")
    ax2.set_title("Probabilidad de P&L final negativo")

    fig.tight_layout()

    os.makedirs(os.path.dirname(guardar_en), exist_ok=True)
    fig.savefig(guardar_en, dpi=150)
    plt.close(fig)
    return guardar_en


def generar_figuras(trades_df, pnl_montecarlo, resumen_montecarlo, directorio=None):
    if directorio is None:
        directorio = _DIRECTORIO_OUTPUT

    os.makedirs(directorio, exist_ok=True)
    return [
        graficar_densidad_y_spreads(os.path.join(directorio, "01_densidad_y_spreads.png")),
        graficar_pnl_por_trade(trades_df, os.path.join(directorio, "02_pnl_por_trade.png")),
        graficar_montecarlo_pnl_final(pnl_montecarlo, os.path.join(directorio, "03_montecarlo_pnl_final.png")),
        graficar_resumen_regimenes(resumen_montecarlo, os.path.join(directorio, "04_resumen_regimenes.png")),
    ]


def graficar_sensibilidad(resultados, ruta_salida=None):
    if ruta_salida is None:
        ruta_salida = os.path.join(_DIRECTORIO_OUTPUT, "05_sensibilidad_pi_i.png")

    pi_i_vals = [r["pi_i"] for r in resultados]
    spreads_num = [r["spread"] for r in resultados]
    spreads_teo = [r["spread_teorico"] for r in resultados]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pi_i_vals, spreads_num, "o-", label="Spread óptimo (numérico)")
    ax.plot(pi_i_vals, spreads_teo, "s--", label="Spread teórico (condición de primer orden)")
    ax.set_xlabel("πᵢ (probabilidad de trader informado)")
    ax.set_ylabel("Spread óptimo (A* − B*)")
    ax.set_title("Sensibilidad del spread óptimo respecto a πᵢ")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida