import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# === CAMINHO DO ARQUIVO DE LOGS ===
LOG_FILE = "monitoring_logs.json"

def persist_data(entry: dict):
    """Persiste dados de monitoramento em JSON (append)."""
    try:
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

        with open(LOG_FILE, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Erro ao salvar log: {e}")


# --- FERRAMENTA PERSONALIZADA (CUSTOM TOOL) ---
class ApiMonitoringTools(BaseTool):
    name: str = "API Monitoring Tool"
    description: str = "Ferramenta para fazer requisições aos endpoints de monitoramento da API."

    def _run(self, endpoint: str) -> str:
        api_base_url = os.getenv("API_BASE_URL")
        api_key = os.getenv("MONITORING_API_KEY")

        if not api_base_url or not api_key:
            return "Erro: Variáveis de ambiente API_BASE_URL ou MONITORING_API_KEY não foram definidas."

        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": api_key
        }

        try:
            url = f"{api_base_url}{endpoint}"
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.Timeout:
            return f"Erro: Timeout ao acessar {endpoint}"
        except requests.exceptions.HTTPError as e:
            return f"Erro HTTP em {endpoint}: {e}"
        except requests.exceptions.RequestException as e:
            return f"Erro geral ao acessar {endpoint}: {e}"


# Instanciando a ferramenta
api_tool = ApiMonitoringTools()


# --- AGENTES ---
data_collector_agent = Agent(
    role='Coletor de Métricas da API',
    goal='Coletar dados vitais dos endpoints da API.',
    backstory='Robô eficiente que coleta dados brutos da API.',
    tools=[api_tool],
    verbose=True
)

data_analyzer_agent = Agent(
    role='Analista de Saúde da Aplicação',
    goal='Analisar os dados JSON coletados e apontar anomalias.',
    backstory='Analista com olhar apurado para detalhes e falhas.',
    verbose=True
)

notification_agent = Agent(
    role='Gerador de Alertas',
    goal='Gerar alertas claros e concisos a partir da análise.',
    backstory='Especialista em comunicação técnica para notificações.',
    verbose=True
)


# --- TAREFAS ---
collect_data_task = Task(
    description=(
        'Chame os endpoints: `/actuator/health` e `/actuator/metrics/jvm.memory.used`. '
        'Compile os resultados em um único output JSON.'
    ),
    expected_output='Dados JSON completos dos endpoints health e jvm.memory.used.',
    agent=data_collector_agent
)

analyze_data_task = Task(
    description=(
        'Analise os dados coletados. '
        'Status deve estar "UP". Memória acima de 700MB = alerta. '
        'Gere relatório curto com conclusões.'
    ),
    expected_output='Relatório conciso com status e uso de memória.',
    agent=data_analyzer_agent,
    context=[collect_data_task]
)

notify_task = Task(
    description=(
        'Com base na análise, crie mensagem final de notificação. '
        'Positiva se tudo estiver ok, alerta caso contrário.'
    ),
    expected_output='Mensagem de notificação final.',
    agent=notification_agent,
    context=[analyze_data_task]
)


# --- MONTAGEM E EXECUÇÃO ---
api_monitoring_crew = Crew(
    agents=[data_collector_agent, data_analyzer_agent, notification_agent],
    tasks=[collect_data_task, analyze_data_task, notify_task],
    process=Process.sequential
)

try:
    result = api_monitoring_crew.kickoff()

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "result": result
    }

    persist_data(log_entry)

    print("\n\n########################")
    print("## Resultado Final do Monitoramento:")
    print("########################\n")
    print(result)
    print("📁 Resultado salvo em monitoring_logs.json")

except Exception as e:
    print("❌ Erro crítico durante execução do Crew:", str(e))
    persist_data({
        "timestamp": datetime.utcnow().isoformat(),
        "error": str(e)
    })