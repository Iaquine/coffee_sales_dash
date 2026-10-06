# ☕ Coffee Retail MLOps: Plataforma de Inteligência de Varejo de Café

Sistema de *machine learning* de ponta a ponta para uma rede varejista de café em crescimento, focado em transformar dados transacionais brutos em inteligência operacional. Construído sob a metodologia CRISP-DM, o projeto conta com rastreabilidade completa de experimentos via MLflow, *pipelines* reprodutíveis e um *dashboard* executivo interativo em Dash.

---

## 📖 Visão Geral do Projeto

### O Problema de Negócio
A empresa acumulava grandes volumes de dados sem os traduzir em decisões, enfrentando dois desafios críticos:
1. **Ineficiência na cadeia de suprimentos:** A dependência de previsões manuais de estoque gerava desperdício de produtos perecíveis e perdas de receita por ruptura de estoque (stockout) de produtos de alta demanda. Estima-se que isto cause uma erosão de margem de aproximadamente 8–13%.
2. **Pontos cegos na retenção de clientes:** A ausência de segmentação baseada em dados impedia a identificação proativa de clientes de alto valor em risco de abandono (*churn*), tornando as campanhas de marketing genéricas e ineficientes.

### A Solução Implementada
Foi desenvolvida uma arquitetura MLOps para inferência em lote (*batch inference*), garantindo baixos custos e imunidade a falhas de API.

| Componente | Função | Artefato Gerado |
|---|---|---|
| **Demand Forecasting** | Prevê a demanda diária por produto usando XGBoost com variáveis temporais (*lag features*). | `demand_forecast.parquet` |
| **RFM Clustering** | Segmenta os clientes em perfis comportamentais através do algoritmo K-Means. | `customer_segments.parquet` |
| **Churn Classification** | Classifica o risco de abandono por cliente de forma supervisionada. | `churn_predictions.parquet` |
| **Monitoring** | Detecta desvios de dados (*data drift* / *concept drift*) e calcula o ROI financeiro. | Métricas no MLflow |
| **Dashboard Executivo** | Interface interativa com simulador de ROI, KPIs e alertas. | `localhost:8050` |

---

## 📊 Dataset e Análise Exploratória (EDA)

Os dados provêm do dataset Kaggle "Coffee Sales" e cobrem cerca de um ano completo de operações (março de 2024 a março de 2025).
* **Ingestão Dupla:** O dataset é composto por `index_1.csv` (transações com identificação de cartão) e `index_2.csv` (pagamentos exclusivamente em dinheiro, sem a coluna `card`).
* **Comportamento de Compra:** O "Latte" lidera a receita absoluta, enquanto o "Americano with Milk" domina o volume de vendas.
* **Padrões Temporais:** O pico de vendas ocorre às 10h da manhã e as terças-feiras registram o maior volume transacional.
* **Identificação:** Cerca de 91% das transações (3.547) estão associadas a clientes com cartão, viabilizando análises de RFM e de *churn*.

---

## 🧠 Fundamentação Teórica: Algoritmos e Métricas

As decisões de engenharia de machine learning foram baseadas rigorosamente nas descobertas da EDA para lidar com as características únicas dos dados de varejo.

### Algoritmos Avaliados
* **XGBoost (eXtreme Gradient Boosting) [Modelo Campeão]:** Escolhido tanto para a previsão de demanda quanto para a classificação de *churn*. Na previsão, foi capaz de capturar não-linearidades e os picos abruptos de vendas diárias ao utilizar engenharias de defasagem (*lags* temporais). No modelo de *churn*, lidou com o forte desbalanceamento de classes nativamente por meio do parâmetro `scale_pos_weight`, evitando a adição de complexidade computacional que técnicas como o SMOTE trariam.
* **Prophet [Descartado]:** Testado para a previsão de demanda, mas rejeitado por retornar um erro muito elevado (MAPE de 97,5%). O Prophet é otimizado para séries temporais com sazonalidades suaves, o que não se aplica à extrema volatilidade diária e ao histórico curto (apenas 1 ano) do varejo de café.
* **K-Means [Modelo Campeão]:** Escolhido para a clusterização de clientes (RFM) operando com K=3. Alcançou o melhor *Silhouette Score* para K≥3, gerando três segmentos comerciais fáceis de serem interpretados e operados pela equipe de marketing (*Champions*, *Loyal* e *Potential*).
* **DBSCAN [Descartado]:** Avaliado na segmentação, mas falhou criticamente ao gerar 98,4% de ruído na classificação. O espaço de variáveis RFM, após a normalização Min-Max, tornou-se extremamente denso. Como o DBSCAN depende da identificação de regiões de baixa densidade para separar clusters, sua aplicação tornou-se inviável neste cenário.

### Métricas de Avaliação Selecionadas
* **MAPE (Mean Absolute Percentage Error):** Métrica primária para a tarefa de Demand Forecasting. Por ser uma taxa percentual, o MAPE garante a comparação justa do desempenho preditivo entre produtos que possuem escalas financeiras e volumes de venda totalmente distintos.
* **PR-AUC (Precision-Recall Area Under the Curve):** Métrica primária de otimização na classificação de *churn*. Ao lidar com bases profundamente desbalanceadas (apenas 25,2% de clientes ativos), o PR-AUC penaliza erros na classe minoritária de forma muito mais severa e realista do que o tradicional ROC-AUC.
* **Silhouette Score:** Métrica guia para o agrupamento não-supervisionado. Ela permite avaliar simultaneamente o quão coesos são os clientes dentro do mesmo cluster e o quão bem separados eles estão dos perfis dos demais clusters.

---

## ⚙️ Arquitetura de Dados e Pipeline

A esteira de processamento segue uma arquitetura MLOps assíncrona estruturada no CRISP-DM.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        CRISP-DM em Camadas                          │
│                                                                     │
│  RAW DATA          PROCESSED          FEATURES          PREDICTIONS │
│  ─────────         ──────────         ────────          ─────────── │
│  index_1.csv  ──►  transactions   ──► demand_feat   ──► demand_fore │
│  index_2.csv       .parquet           .parquet          cast.parquet│
│                        │          ──► rfm_feat          churn_pred_ │
│                        │              .parquet          ictions.par │
│                        │          ──► churn_feat        quet        │
│                        │              .parquet          customer_se │
│                        │          ──► customer_se       gments.par  │
│                        │              gments.par        quet        │
│                        │              quet                          │
│                                                                     │
│  ┌─────────────────────▼─────────────────────────────────────────┐  │
│  │                   MLflow Tracking                             │  │
│  │  demand_forecasting | rfm_clustering | churn_classification   │  │
│  │  monitoring                                                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                        │                                            │
│  ┌─────────────────────▼─────────────────────────────────────────┐  │
│  │               Dashboard (Dash / Plotly)                       │  │
│  │  Aba 1: KPIs  │  Aba 2: Demanda  │  Aba 3: Churn  │ Aba 4: ROI│  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Observabilidade MLOps e Cálculo de ROI
* **Desvio de Dados (*Data Drift*):** O pipeline utiliza o Teste KS (Kolmogorov-Smirnov) e o PSI (*Population Stability Index*) para monitoramento. No cenário simulado atual, foi detectado um desvio crítico na variável `rolling_mean_7` (PSI > 0.2), indicando mudanças de comportamento temporal que acionam automaticamente o script de re-treinamento.
* **Benefício Financeiro:** Subtraindo custos operacionais de nuvem e campanhas, o simulador do painel computou um Benefício Líquido de R$ 2.704 e um ROI projetado de 310,9%, decorrentes da redução no desperdício de perecíveis e direcionamento de marketing de precisão.

---

## 🛠️ Instalação e Pré-requisitos

* **Python:** 3.10 ou superior.
* **Gerenciador de pacotes:** `pip` (ou conda).
* **Dependências principais:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `mlflow`, `dash`, `plotly`, `pandera`, `pytest`, `pyarrow`.

```bash
# 1. Clonar o repositório ou navegar para o diretório
cd coffee_sales/

# 2. Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Instalar dependências
pip install -r requirements.txt
```

### Download do Dataset
O dataset pode ser baixado diretamente do Kaggle:
```bash
# Utilizando a CLI do Kaggle
pip install kaggle
kaggle datasets download -d ihelon/coffee-sales -p data/raw/ --unzip
```
*(Certifique-se de que os arquivos `index_1.csv` e `index_2.csv` fiquem descompactados na pasta `data/raw/`)*.

---

## 🚀 Execução

Todos os comandos devem ser executados a partir do diretório raiz `coffee_sales/`.

**1. Pipeline Completo (Treinamento end-to-end)**
Executa a ingestão, pré-processamento, criação de features, treinamento e monitoramento:
```bash
python pipelines/training_pipeline.py
```

**2. Pipeline sem Re-treino**
Atualiza as variáveis e emite relatório de desvios (*drift*) sem executar um novo treinamento dos algoritmos:
```bash
python pipelines/training_pipeline.py --skip-training
```

*(Opcional: Indicar diretório alternativo de dados brutos)*:
```bash
python pipelines/training_pipeline.py --raw-dir /caminho/para/raw
```

**3. Testes Unitários e de Integração**
O projeto garante a robustez das operações através de uma suíte com 163 testes.
```bash
pytest tests/ -v
```

**4. Iniciar o Dashboard Executivo**
```bash
python src/dashboard/app.py
# Acessar em: http://localhost:8050
```

**5. Iniciar o MLflow UI**
```bash
mlflow ui --backend-store-uri mlruns/
# Acessar em: http://localhost:5000
```

**6. Simular Re-treino Automático**
Dispara o ciclo de aprendizado caso os alertas do drift detector estejam críticos:
```bash
python pipelines/retraining_pipeline.py
```

---

## 📁 Estrutura Completa do Projeto

```text
coffee_sales/
├── data/
│   ├── raw/                        # Arquivos originais Kaggle (index_1.csv, index_2.csv)
│   ├── processed/                  # Dados limpos e padronizados (transactions.parquet)
│   ├── features/                   # Base engenheirada (RFM, Lags Temporais e Churn labels)
│   └── predictions/                # Resultados e previsões (arquivos parquet lidos pelo Dash)
├── notebooks/                      # Jupyter Notebooks de EDA e explorações preliminares
├── src/
│   ├── data/                       # Scripts modulares de ingestão e pré-processamento
│   ├── features/                   # Fábrica de variáveis com proteção contra data leakage
│   ├── models/                     # Rotinas de treino otimizadas (forecasting, clustering, churn)
│   ├── monitoring/                 # Detecção de drift (PSI/KS) e calculadoras de ROI dinâmico
│   └── dashboard/                  # Aplicação modular executiva em Dash/Plotly
├── pipelines/                      # Orquestradores principais do MLOps
├── tests/                          # Suíte completa de testes Pytest (163 testes independentes)
├── mlruns/                         # Registro local de experimentos e versões do MLflow
├── requirements.txt                # Lista de dependências Python
└── README.md                       # Este arquivo
```

---

## 🔗 Referências e Roadmap de Evolução
No curto prazo, a prioridade é aplicar Testes A/B rigorosos em grupos de controle nas campanhas de retenção. A médio e longo prazo, a meta é agregar dados climáticos para enriquecer as previsões e migrar a orquestração do pipeline para nuvem usando o Apache Airflow.

* **Dataset Base:** [Kaggle — Coffee Sales](https://www.kaggle.com/datasets/ihelon/coffee-sales).
* **Tecnologias:** [MLflow](https://mlflow.org), [XGBoost](https://xgboost.readthedocs.io), [Dash/Plotly](https://dash.plotly.com).