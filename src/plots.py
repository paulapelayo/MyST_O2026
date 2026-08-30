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

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import erlang

from src import model
from src import simulation

REGIMEN_ORDEN = ["optimo", "estrecho", "amplio"]
REGIMEN_ETIQUETAS = {"optimo": "Óptimo", "estrecho": "Estrecho", "amplio": "Amplio"}
REGIMEN_COLORES = {"optimo": "#2ca02c", "estrecho": "#d62728", "amplio": "#1f77b4"}


def graficar_densidad_y_spreads(guardar_en="output/01_densidad_y_spreads.png"):
    """
    Grafica la densidad Erlang del valor verdadero del activo (f(P) de
    src/model.py) y marca, con líneas verticales, el Bid y el Ask de cada
    régimen de cotización. Sirve para ver visualmente qué tan "adentro" o
    "afuera" de la distribución queda cada spread.
    """
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


def graficar_pnl_por_trade(trades_df, guardar_en="output/02_pnl_por_trade.png"):
    """
    Grafica, en un panel por régimen, el histograma del P&L de cada trade
    simulado. Los intentos que no se ejecutaron (P&L = 0 por definición,
    ver simulation._generar_pnl) se excluyen del histograma para no
    aplastar la escala con un solo pico en cero; en su lugar, cada panel
    anota qué porcentaje de los intentos no se ejecutó. trades_df es el
    DataFrame que regresa simulation.simular_todos_los_regimenes().
    """
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


def graficar_montecarlo_pnl_final(pnl_por_regimen, guardar_en="output/03_montecarlo_pnl_final.png"):
    """
    Grafica, sobrepuestos en un solo panel, los histogramas del P&L final
    (acumulado en 1,000 trades) de las 1,000 corridas de Monte Carlo de
    cada régimen. pnl_por_regimen es el diccionario que regresa
    simulation.monte_carlo_pnl_por_regimen().
    """
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


def graficar_resumen_regimenes(resumen_df, guardar_en="output/04_resumen_regimenes.png"):
    """
    Grafica un resumen comparativo de los tres regímenes: P&L promedio
    con barras de error de una desviación estándar, y probabilidad de
    pérdida. resumen_df es el DataFrame que regresa
    simulation.resumir_montecarlo() (o monte_carlo_regimenes()).
    """
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


def generar_figuras(trades_df, pnl_montecarlo, resumen_montecarlo, directorio="output"):
    """
    Genera y guarda las cuatro figuras del laboratorio a partir de
    resultados ya calculados (no vuelve a simular). Regresa la lista de
    rutas de los archivos guardados.
    """
    os.makedirs(directorio, exist_ok=True)
    return [
        graficar_densidad_y_spreads(f"{directorio}/01_densidad_y_spreads.png"),
        graficar_pnl_por_trade(trades_df, f"{directorio}/02_pnl_por_trade.png"),
        graficar_montecarlo_pnl_final(pnl_montecarlo, f"{directorio}/03_montecarlo_pnl_final.png"),
        graficar_resumen_regimenes(resumen_montecarlo, f"{directorio}/04_resumen_regimenes.png"),
    ]
