import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
from langchain.chat_models import ChatOpenAI  # Integração com LLM via RouteLLM

# Carrega as variáveis de ambiente
load_dotenv()

# Configuração do RouteLLM (Abacus.AI)
llm = ChatOpenAI(
    model="gpt-5-mini",  # Você pode trocar p/ outra LLM disponível no RouteLLM
    openai_api_base="https://api.abacus.ai/llm/v1",
    api_key=os.getenv("ABACUS_API_KEY"),  # chave única da Abacus.AI
    temperature=0
)

# === PERSISTÊNCIA DE LOGS ===
LOG_FILE = "monitoring_logs.json"

def persist_data(entry: dict):
    """Salva os resultados em JSON (append)."""
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


# --- FERRAMENTA CUSTOM ---
class ApiMonitoringTools(BaseTool):
    name: str = "API Monitoring Tool"
    description: str = "Ferramenta para requisições aos endpoints de monitoramento da API."

    def _run(self, endpoint: str) -> str:
        api_base_url = os.getenv("API_BASE_URL")
        api_key = os.getenv("MONITORING_API_KEY")

        if not api_base_url or not api_key:
            return "Erro: Variáveis API_BASE_URL ou MONITORING_API_KEY não definidas."

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

# Instância da ferramenta
api_tool = ApiMonitoringTools()


# --- AGENTES (usando RouteLLM) ---
data_collector_agent = Agent(
    role='Coletor de Métricas da API',
    goal='Coletar dados vitais dos endpoints de saúde.',
    backstory='Robô especializado em requisições HTTP para métricas.',
    tools=[api_tool],
    verbose=True,
    llm=llm
)

data_analyzer_agent = Agent(
    role='Analista de Saúde da Aplicação',
    goal='Interpretar os dados coletados e detectar anomalias.',
    backstory='Especialista em identificação de falhas em sistemas.',
    verbose=True,
    llm=llm
)

notification_agent = Agent(
    role='Gerador de Alertas',
    goal='Transformar os insights técnicos em alertas claros e objetivos.',
    backstory='Profissional em comunicação técnica para times de dev.',
    verbose=True,
    llm=llm
)


# --- TAREFAS ---
collect_data_task = Task(
    description=(
        'Busque dados em dois endpoints: `/actuator/health` e `/actuator/metrics/jvm.memory.used`. '
        'Combine a resposta em JSON unificado.'
    ),
    expected_output='JSON com dados brutos health + jvm.memory.used.',
    agent=data_collector_agent
)

analyze_data_task = Task(
    description=(
        'Analise os dados: health deve estar UP. '
        'Se memória usada > 700MB → levantar alerta. '
        'Produza resumo curto e objetivo.'
    ),
    expected_output='Relatório com status e uso de memória.',
    agent=data_analyzer_agent,
    context=[collect_data_task]
)

notify_task = Task(
    description=(
        'Com base na análise, redija a mensagem final de notificação. '
        'Se tudo ok → mensagem positiva. '
        'Se falha → alerta claro e conciso.'
    ),
    expected_output='Mensagem final para equipe.',
    agent=notification_agent,
    context=[analyze_data_task]
)


# --- CREW ---
api_monitoring_crew = Crew(
    agents=[data_collector_agent, data_analyzer_agent, notification_agent],
    tasks=[collect_data_task, analyze_data_task, notify_task],
    process=Process.sequential
)

try:
    result = api_monitoring_crew.kickoff()

    log_entry = {"timestamp": datetime.utcnow().isoformat(), "result": result}
    persist_data(log_entry)

    print("\n\n########################")
    print("## Resultado Final do Monitoramento:")
    print("########################\n")
    print(result)
    print("📁 Salvo em monitoring_logs.json")

except Exception as e:
    print("❌ Erro durante execução:", str(e))
    persist_data({
        "timestamp": datetime.utcnow().isoformat(),
        "error": str(e)
    })