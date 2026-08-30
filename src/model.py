"""
Modelo de Copeland y Galai (1983): función de utilidad del market maker y
optimización del spread bid-ask.

El formador de mercado (market maker) no sabe si la próxima orden que recibe
viene de un trader informado (que conoce el valor verdadero del activo) o de
un trader de liquidez (que opera por razones ajenas a la información). Por
eso fija un precio de compra Bid (B) y un precio de venta Ask (A) alrededor
de un precio de referencia S0, buscando maximizar su utilidad esperada por
trade dado ese riesgo.
"""

import numpy as np
from scipy import integrate, optimize
from scipy.stats import erlang


# ---------------------------------------------------------------------------
# Parámetros del caso base
# ---------------------------------------------------------------------------

S0 = 19.90       # Precio de referencia del activo
K = 60           # Parámetro de forma (shape) de la distribución Erlang
LAMBDA = 3       # Parámetro de tasa (rate) de la distribución Erlang
PI_I = 0.40      # Probabilidad de que la contraparte sea un trader informado
PI_L = 0.60      # Probabilidad de que la contraparte sea un trader de liquidez


def f(P):
    """
    Densidad Erlang(K, λ) del valor verdadero del activo.

    Representa, según toda la información disponible en el mercado, qué tan
    probable es que el valor "real" del activo sea P. El market maker usa
    esta densidad para estimar cuánto puede perder si un trader informado
    (que sí conoce ese valor real) opera contra su Bid o su Ask.
    """
    return erlang.pdf(P, K, scale=1 / LAMBDA)


def pi_LB(s):
    """
    Probabilidad de ejecución de un trader de liquidez al precio Bid.

    s es la distancia entre el precio pactado y el precio de referencia S0.
    Entre más se aleje el market maker de S0, menos atractivo resulta el
    precio para el trader de liquidez, así que la probabilidad de que
    decida ejecutar baja linealmente hasta un mínimo de cero (nunca
    negativa).
    """
    return max(0.50 - 0.08 * s, 0.0)


def pi_LS(s):
    """
    Probabilidad de ejecución de un trader de liquidez al precio Ask.

    Misma lógica que pi_LB: s es la distancia respecto a S0 y la
    probabilidad de ejecución baja linealmente conforme el precio se aleja
    de la referencia, acotada en cero por abajo. En este modelo pi_LB y
    pi_LS son la misma función.
    """
    return max(0.50 - 0.08 * s, 0.0)


def perdida_esperada_ask(A):
    """
    Pérdida esperada del market maker al vender en el Ask (A).

    Calcula ∫_A^∞ (P − A)·f(P) dP: en promedio, cuánto pierde el market
    maker cuando vende en A a un trader informado que sabe que el activo
    en realidad vale más (P > A), ponderando cada posible valor P por qué
    tan probable es según f(P).
    """
    valor, _ = integrate.quad(lambda P: (P - A) * f(P), A, np.inf)
    return valor


def perdida_esperada_bid(B):
    """
    Pérdida esperada del market maker al comprar en el Bid (B).

    Calcula ∫_0^B (B − P)·f(P) dP: en promedio, cuánto pierde el market
    maker cuando compra en B a un trader informado que sabe que el activo
    en realidad vale menos (P < B), ponderando cada posible valor P por
    qué tan probable es según f(P).
    """
    valor, _ = integrate.quad(lambda P: (B - P) * f(P), 0, B)
    return valor


def utilidad_esperada(A, B):
    """
    Utilidad esperada por trade del market maker, Π(A, B).

    Es la ganancia esperada de operar con traders de liquidez (que pagan
    el spread, ponderada por su probabilidad de ejecución) menos la
    pérdida esperada de operar con traders informados (que siempre
    aprovechan la diferencia entre el precio pactado y el valor real del
    activo). El market maker elige A y B para maximizar esta cantidad.
    """
    ganancia_liquidez = PI_L * (
        pi_LB(A - S0) * (A - S0) + pi_LS(S0 - B) * (S0 - B)
    )
    perdida_informados = PI_I * (
        perdida_esperada_ask(A) + perdida_esperada_bid(B)
    )
    return ganancia_liquidez - perdida_informados


def _objetivo(x):
    """
    Función objetivo para el optimizador: el negativo de Π(A, B).

    scipy.optimize.minimize solo minimiza funciones, así que para
    encontrar el A y B que maximizan la utilidad esperada del market
    maker, se minimiza -Π(A, B) en su lugar.
    """
    A, B = x
    return -utilidad_esperada(A, B)


def optimizar_spread(x0=None):
    """
    Calcula el Bid y Ask óptimos que maximizan Π(A, B).

    Resuelve el problema de optimización del market maker sujeto a
    B ∈ (0, S0] y A ∈ [S0, ∞). Regresa el Bid óptimo, el Ask óptimo, el
    spread resultante (Ask − Bid) y la utilidad esperada por trade,
    todos redondeados a 2 decimales.
    """
    if x0 is None:
        x0 = [S0 + 0.10, S0 - 0.10]

    limites = [(S0, None), (1e-6, S0)]  # (A, B) respectivamente

    resultado = optimize.minimize(
        _objetivo, x0=x0, method="L-BFGS-B", bounds=limites
    )

    A_opt, B_opt = resultado.x
    utilidad_opt = utilidad_esperada(A_opt, B_opt)

    return {
        "bid": round(float(B_opt), 2),
        "ask": round(float(A_opt), 2),
        "spread": round(float(A_opt - B_opt), 2),
        "utilidad_esperada": round(float(utilidad_opt), 2),
    }


if __name__ == "__main__":
    print(optimizar_spread())
