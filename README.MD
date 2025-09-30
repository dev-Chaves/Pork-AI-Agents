# 🐷 Pork-AI-Agents: Monitoramento Inteligente com Agentes de IA

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![CrewAI](https://img.shields.io/badge/CrewAI-Agentic%20Framework-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

Este projeto implementa um sistema de monitoramento autônomo para aplicações (especialmente Spring Boot com Actuator), utilizando uma equipe de agentes de IA orquestrada pelo framework **CrewAI**. A solução se conecta a um Large Language Model (LLM) através do **RouteLLM da Abacus.AI**, permitindo uma análise inteligente e contextual das métricas da aplicação.

Além do monitoramento agendado, o sistema integra-se a um bot do **Telegram** para enviar alertas e receber comandos, tornando a operação mais interativa e acessível.

---

## 📜 Tabela de Conteúdos

1.  [Como Funciona: A Equipe de Agentes](#-como-funciona-a-equipe-de-agentes-crew)
2.  [Principais Funcionalidades](#-principais-funcionalidades)
3.  [Estrutura do Projeto](#-estrutura-do-projeto)
4.  [Como Começar](#️-como-começar)
    * [Pré-requisitos](#pré-requisitos)
    * [Configuração do Ambiente](#1-configuração-do-ambiente)
    * [Executando com Docker](#2-executando-com-docker-recomendado)
    * [Executando Localmente](#3-executando-localmente)
5.  [Tecnologias Utilizadas](#️-tecnologias-utilizadas)
6.  [Melhorias Futuras](#-melhorias-futuras)

---

## 🚀 Como Funciona: A Equipe de Agentes (Crew)

O coração do sistema é uma equipe de três agentes de IA, cada um com uma função específica, que trabalham em sequência para transformar dados brutos em insights acionáveis.

### 1. 🤖 Agente Coletor de Dados (`Data Collector Agent`)
- **Missão:** Acessar os endpoints do Spring Boot Actuator da aplicação-alvo.
- **Ferramentas:** `ApiMonitoringTool` (uma ferramenta customizada que faz requisições HTTP com retentativas).
- **Resultado:** Agrupa todas as informações coletadas (saúde, métricas de CPU/memória, traces HTTP, etc.) em um único documento JSON estruturado.

### 2. 🧠 Agente Analista de Dados (`Data Analyzer Agent`)
- **Missão:** Interpretar o JSON bruto fornecido pelo Coletor. Ele converte métricas (ex: bytes para MB), calcula percentuais, e compara os valores com *thresholds* (limites) pré-definidos (SLOs).
- **Inteligência:** Utiliza o LLM via RouteLLM para identificar anomalias, correlacionar eventos e gerar um diagnóstico completo.
- **Resultado:** Produz um novo JSON com uma análise aprofundada, incluindo nível de severidade (`INFO`, `WARN`, `ALERT`), *findings* detalhados e ações recomendadas.

### 3. 📢 Agente Notificador (`Notification Agent`)
- **Missão:** Traduzir a análise técnica e estruturada do Analista em uma mensagem clara, concisa e humana.
- **Comunicação:** Formata o alerta para ser facilmente compreendido pela equipe de desenvolvimento ou SRE.
- **Resultado:** Gera a mensagem final que será enviada para o Telegram.

---

## ✨ Principais Funcionalidades

- **Monitoramento Automatizado:** O script roda em um agendamento (diariamente às 12:00, por padrão) para verificações periódicas.
- **Análise Potencializada por IA:** Vai além de simples alertas de "CPU alta", fornecendo contexto e sugestões de investigação.
- **Controle Interativo via Telegram:** Envie comandos para o bot para interagir com o sistema:
  - `/run`: Dispara uma execução de monitoramento imediatamente.
  - `/status`: Verifica se o monitor está online.
  - `/logs`: Exibe o resultado da última execução.
  - `/config`: Mostra os thresholds de alerta atuais.
  - `/help`: Lista todos os comandos.
- **Containerizado e Pronto para Deploy:** Com `Dockerfile` e `compose.yaml`, a implantação é simples e isolada.
- **Persistência de Logs:** Cada execução é salva em um arquivo `monitoring_logs.json`.

---

## 📂 Estrutura do Projeto

├── .dockerignore         # Arquivos a serem ignorados pela imagem Docker.
├── .env                  # Arquivo para variáveis de ambiente (NÃO versionado).
├── .gitignore            # Arquivos a serem ignorados pelo Git.
├── Dockerfile            # Define a imagem Docker para rodar o monitor.
├── README.MD             # Este arquivo.
├── compose.yaml          # Orquestra o container do monitor com Docker Compose.
├── monitor.py            # O script principal que define e executa os agentes.
├── monitoring_logs.json  # (Gerado) Logs das execuções do monitor.
└── requirements.txt      # Dependências Python do projeto.

## 🛠️ Como Começar

### Pré-requisitos

- Python 3.9+
- Docker e Docker Compose (recomendado)
- Um Bot do Telegram e o ID de um Chat para receber as notificações.
- Acesso a um serviço de LLM (configurado via RouteLLM).

### 1. Configuração do Ambiente

Clone o repositório e crie um arquivo `.env` na raiz do projeto. Use o exemplo abaixo como base:

```env
# Chave de API para o RouteLLM (ou o provedor de LLM que ele roteia)
OPENAI_API_KEY="sk-..."

# URL base da API a ser monitorada (com endpoints Actuator)
API_BASE_URL="[http://sua-api.example.com](http://sua-api.example.com)"

# Chave de API para autenticar nos endpoints de monitoramento (X-API-KEY)
MONITORING_API_KEY="sua-chave-de-monitoramento"

# --- Telegram (Opcional, mas recomendado) ---
TELEGRAM_BOT_TOKEN="token-do-seu-bot"
TELEGRAM_CHAT_ID="id-do-seu-chat"

# --- Thresholds de Alerta (Opcional) ---
MEMORY_ALERT_MB=700
CPU_ALERT_PCT=85.0
ERROR_RATE_SLO_PCT=1.0
LATENCY_P95_SLO_MS=800

# Constrói a imagem e sobe o container em background
docker-compose up --build -d

# Para ver os logs em tempo real
docker-compose logs -f

# Para parar o container
docker-compose down

# 1. Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # No Linux/macOS
# .\.venv\Scripts\Activate.ps1 # No Windows (PowerShell)

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute o script
python monitor.py

