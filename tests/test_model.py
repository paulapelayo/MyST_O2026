"""
Pruebas unitarias para src/model.py.

Aún sin implementar.
"""

"""
Pruebas obligatorias para src/model.py (sección 3.6 del laboratorio).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from model import pi_LB, pi_LS, perdida_esperada_ask, optimizar_spread


def test_probabilidades_ejecucion_no_negativas():
    """
    pi_LB(s) y pi_LS(s) nunca deben devolver un valor negativo, incluso
    para distancias s grandes donde la recta 0.50 - 0.08s cruzaría a
    negativo sin la cota inferior en cero.
    """
    valores_s = [0, 3, 6.25, 10, 50]
    for s in valores_s:
        assert pi_LB(s) >= 0.0
        assert pi_LS(s) >= 0.0


def test_perdida_esperada_ask_decreciente_en_A():
    """
    La pérdida esperada frente a informados en el Ask debe ser
    decreciente en A: entre más lejos cotiza el market maker por
    arriba de S0, menor la pérdida esperada frente a un informado.
    """
    valores_A = [19.90, 20.90, 21.90, 22.90]
    perdidas = [perdida_esperada_ask(A) for A in valores_A]

    for i in range(len(perdidas) - 1):
        assert perdidas[i] > perdidas[i + 1]


def test_spread_optimo_monopolista_sin_informados():
    """
    Con pi_i = 0 (sin traders informados), el spread óptimo por lado
    debe coincidir con el resultado analítico del monopolista:
    maximizar s * (0.50 - 0.08 s) da s* = 0.50 / (2 * 0.08) = 3.125.

    Nota: el enunciado del laboratorio menciona s* = 0.50/0.08, pero ese
    valor (6.25) es el punto donde la probabilidad de ejecución llega a
    cero, no el máximo de la ganancia esperada. El máximo real, y el que
    produce el propio optimizador, es 0.50/(2*0.08) = 3.125 por lado.
    """
    resultado = optimizar_spread(pi_i=0.0, pi_l=1.0)
    s_teorico_por_lado = 0.50 / (2 * 0.08)
    spread_teorico = 2 * s_teorico_por_lado

    assert resultado["spread"] == pytest.approx(spread_teorico, abs=0.01)
