🐷 Pork-AI-Agents: Monitoramento Inteligente com Agentes de IA

Este projeto implementa um sistema de monitoramento autônomo para aplicações (especialmente Spring Boot com Actuator), utilizando uma equipe de agentes de IA orquestrada pelo framework CrewAI.

A solução se conecta a um Large Language Model (LLM) através do RouteLLM da Abacus.AI, permitindo uma análise inteligente e contextual das métricas da aplicação.

Além do monitoramento agendado, o sistema integra-se a um bot do Telegram para enviar alertas e receber comandos, tornando a operação mais interativa e acessível.

📜 Tabela de Conteúdos

Como Funciona: A Equipe de Agentes

Principais Funcionalidades

Estrutura do Projeto

Como Começar

Pré-requisitos

Configuração do Ambiente

Executando com Docker

Executando Localmente

Tecnologias Utilizadas

Melhorias Futuras

🚀 Como Funciona: A Equipe de Agentes (Crew)

O coração do sistema é uma equipe de três agentes de IA, cada um com uma função específica, que trabalham em sequência para transformar dados brutos em insights acionáveis.

1. 🤖 Agente Coletor de Dados (Data Collector Agent)

Missão: Acessar os endpoints do Spring Boot Actuator da aplicação-alvo.

Ferramentas: ApiMonitoringTool (faz requisições HTTP com retentativas).

Resultado: Agrupa informações (saúde, métricas de CPU/memória, traces, etc.) em um JSON estruturado.

2. 🧠 Agente Analista de Dados (Data Analyzer Agent)

Missão: Interpretar o JSON bruto fornecido pelo Coletor.

Inteligência: Utiliza o LLM via RouteLLM para:

Converter métricas (ex: bytes → MB)

Comparar valores com thresholds (SLOs)

Identificar anomalias, gerar diagnóstico completo

Resultado: JSON de análise com severidade (INFO, WARN, ALERT), findings detalhados e ações recomendadas.

3. 📢 Agente Notificador (Notification Agent)

Missão: Traduzir a análise em mensagem clara e humana.

Comunicação: Mensagem formatada para Telegram.

Resultado: Alerta final compreensível pela equipe de desenvolvimento ou SRE.

✨ Principais Funcionalidades

Monitoramento Automatizado: Execuções agendadas (ex.: diariamente às 12:00).

Análise Potencializada por IA: Diagnósticos com contexto, não só alertas crus.

Controle via Telegram:

/run → Executa monitoramento imediato

/status → Verifica se o sistema está online

/logs → Retorna logs da última execução

/config → Mostra thresholds atuais

/help → Lista comandos

Containerizado: Fácil deploy com Docker e Compose.

Persistência de Logs: Saída salva em monitoring_logs.json.

📂 Estrutura do Projeto
├── .dockerignore         # Arquivos ignorados no Docker
├── .env                  # Variáveis de ambiente
├── .gitignore            # Arquivos ignorados no Git
├── Dockerfile            # Imagem Docker do monitor
├── README.MD             # Documentação
├── compose.yaml          # Orquestração com Docker Compose
├── monitor.py            # Script principal (agentes)
├── monitoring_logs.json  # Logs das execuções
└── requirements.txt      # Dependências Python

🛠️ Como Começar
Pré-requisitos

Python 3.9+

Docker e Docker Compose

Bot do Telegram + Chat ID

Chave de API para algum LLM (via RouteLLM)

1. Configuração do Ambiente

Crie um .env na raiz do projeto:

OPENAI_API_KEY="sk-..."
API_BASE_URL="http://sua-api.example.com"
MONITORING_API_KEY="sua-chave-de-monitoramento"

# Telegram (opcional)
TELEGRAM_BOT_TOKEN="token-do-seu-bot"
TELEGRAM_CHAT_ID="id-do-seu-chat"

# Thresholds
MEMORY_ALERT_MB=700
CPU_ALERT_PCT=85.0
ERROR_RATE_SLO_PCT=1.0
LATENCY_P95_SLO_MS=800

2. Executando com Docker (Recomendado)
docker-compose up --build -d    # Sobe container
docker-compose logs -f          # Logs em tempo real
docker-compose down             # Para o container

3. Executando Localmente
# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows

# Instalar dependências
pip install -r requirements.txt

# Rodar monitor
python monitor.py

⚙️ Tecnologias Utilizadas

CrewAI → Orquestração de agentes de IA

RouteLLM (Abacus.AI) → Roteamento inteligente de LLMs

OpenAI Python Client → Comunicação com API

Docker → Containerização

Schedule → Agendamento de tarefas

python-dotenv → Variables de ambiente

💡 Melhorias Futuras

Logging mais robusto (com rotação de arquivos).

Novas ferramentas de agentes (ex.: consultar doc da API).

Criar testes unitários para ApiMonitoringTool.

Suporte a múltiplos alvos de monitoramento.