# Laboratorio 01: Modelo de Copeland y Galai (1983)

Laboratorio de la materia **Microestructura de Mercado** (ITESO) sobre el
modelo de asimetría de información de Copeland y Galai (1983), en el que el
formador de mercado (market maker) fija spreads bid-ask para protegerse del
riesgo de operar contra agentes informados.

## Estructura del proyecto

```
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── src/
│   ├── model.py         # función de utilidad y optimización
│   ├── simulation.py    # simulador de trades
│   └── plots.py         # generación de figuras
├── tests/
│   └── test_model.py
├── notebooks/
│   └── analysis.ipynb   # análisis y figuras, importando funciones de src/
└── docs/                # material de la presentación (PDF)
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

## Estado

Proyecto en configuración inicial. Aún no se ha implementado la lógica del
modelo.
