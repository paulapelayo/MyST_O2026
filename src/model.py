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
from scipy.optimize import brentq

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


def utilidad_esperada(A, B, pi_i=PI_I, pi_l=PI_L):
    """
    Utilidad esperada por trade del market maker, Π(A, B).

    pi_i y pi_l son ahora parámetros (no solo constantes globales) para
    poder reoptimizar el modelo bajo distintos escenarios de probabilidad
    de trader informado, como pide la sección 3.5.
    """
    ganancia_liquidez = pi_l * (
        pi_LB(A - S0) * (A - S0) + pi_LS(S0 - B) * (S0 - B)
    )
    perdida_informados = pi_i * (
        perdida_esperada_ask(A) + perdida_esperada_bid(B)
    )
    return ganancia_liquidez - perdida_informados


def _objetivo(x, pi_i=PI_I, pi_l=PI_L):
    A, B = x
    return -utilidad_esperada(A, B, pi_i=pi_i, pi_l=pi_l)


def optimizar_spread(x0=None, pi_i=PI_I, pi_l=PI_L):
    """
    Calcula el Bid y Ask óptimos que maximizan Π(A, B) para una
    probabilidad de trader informado pi_i dada (por defecto, el caso base).
    """
    if x0 is None:
        x0 = [S0 + 0.10, S0 - 0.10]

    limites = [(S0, None), (1e-6, S0)]

    resultado = optimize.minimize(
        _objetivo, x0=x0, args=(pi_i, pi_l), method="L-BFGS-B", bounds=limites
    )

    A_opt, B_opt = resultado.x
    utilidad_opt = utilidad_esperada(A_opt, B_opt, pi_i=pi_i, pi_l=pi_l)

    return {
        "bid": round(float(B_opt), 2),
        "ask": round(float(A_opt), 2),
        "spread": round(float(A_opt - B_opt), 2),
        "utilidad_esperada": round(float(utilidad_opt), 2),
    }

def f_acumulada(P):
    """CDF Erlang(K, λ): P(valor verdadero <= P)."""
    return erlang.cdf(P, K, scale=1 / LAMBDA)


def spread_teorico(pi_i, pi_l):
    """
    Spread óptimo por lado según la condición de primer orden del modelo:

        π_L·(0.50 − 0.16·s) + π_I·(1 − F(S0 + s)) = 0
    """
    def foc(s):
        A = S0 + s
        F_A = f_acumulada(A)
        return pi_l * (0.50 - 0.16 * s) + pi_i * (1 - F_A)

    return brentq(foc, 0.01, 6.24)


def analisis_sensibilidad(valores_pi_i=(0.1, 0.4, 0.7)):
    """
    Reoptimiza el spread para distintos valores de pi_i, manteniendo
    pi_l = 1 - pi_i, y lo compara contra el spread teórico.
    """
    resultados = []
    for pi_i in valores_pi_i:
        pi_l = 1 - pi_i
        r = optimizar_spread(pi_i=pi_i, pi_l=pi_l)
        r["pi_i"] = pi_i
        r["spread_teorico"] = round(2 * spread_teorico(pi_i, pi_l), 2)
        resultados.append(r)
    return resultados


if __name__ == "__main__":
    print(optimizar_spread())
    print(analisis_sensibilidad())