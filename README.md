# Traffic Prediction with Spatiotemporal Graph Neural Networks

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Proyecto de predicción de congestión de tráfico utilizando Redes Neuronales de Grafos Espaciotemporales (STGNN) basado en [Torch Spatiotemporal (tsl)](https://torch-spatiotemporal.readthedocs.io/).

## 📋 Descripción

Este proyecto implementa un modelo de aprendizaje profundo para predecir el tráfico vehicular utilizando datos de sensores distribuidos en una red de carreteras. El modelo aprovecha tanto las relaciones temporales (patrones de tráfico a lo largo del tiempo) como las relaciones espaciales (conectividad entre sensores en la red vial).

### Características Principales

- 🧠 **Arquitectura Time-then-Space**: Procesa primero patrones temporales con RNN, luego patrones espaciales con GNN
- 📊 **Dataset MetrLA**: 207 sensores de tráfico en autopistas de Los Ángeles
- ⚡ **PyTorch Lightning**: Entrenamiento estructurado y escalable
- 📈 **TensorBoard**: Monitoreo en tiempo real del entrenamiento
- 🔧 **Configuración YAML**: Fácil ajuste de hiperparámetros
- 📦 **Código Modular**: Estructura clara y mantenible
- ✅ **Tests Unitarios**: Cobertura de componentes críticos

## 🏗️ Arquitectura del Modelo