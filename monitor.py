import os
import json
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool  # Import atualizado do CrewAI Tools
from langchain_openai import ChatOpenAI  # LangChain OpenAI moderno

# Carrega as variáveis de ambiente (.env)
load_dotenv()

# Remover variáveis que possam conflitar com o uso do RouteLLM da Abacus
for var in ["OPENAI_BASE_URL", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "LITELLM_BASE_URL"]:
    if os.environ.get(var):
        os.environ.pop(var)

abacus_key = os.getenv("ABACUS_API_KEY")
if not abacus_key:
    raise RuntimeError("ABACUS_API_KEY não definido no .env")

# Compatibilidade com wrappers que exigem OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = abacus_key

# =========================================
# Configuração do RouteLLM (Abacus.AI)
# =========================================
llm = ChatOpenAI(
    model="gpt-5-mini",
    base_url="https://api.abacus.ai/llm/v1",  # essencial
    api_key=abacus_key,                       # essencial
    temperature=0,
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
            f.truncate()
    except Exception as e:
        print(f"⚠️ Erro ao salvar log: {e}")


# --- FERRAMENTA CUSTOM ---
class ApiMonitoringTool(BaseTool):
    """
    Ferramenta para requisições aos endpoints de monitoramento da API.
    Usa API_BASE_URL e MONITORING_API_KEY do .env.
    """
    name: str = "API Monitoring Tool"
    description: str = "Faz GET em endpoints de monitoramento e retorna o texto da resposta."

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
            return response.text  # ou: json.dumps(response.json(), ensure_ascii=False)
        except requests.exceptions.Timeout:
            return f"Erro: Timeout ao acessar {endpoint}"
        except requests.exceptions.HTTPError as e:
            return f"Erro HTTP em {endpoint}: {e}"
        except requests.exceptions.RequestException as e:
            return f"Erro geral ao acessar {endpoint}: {e}"

# Instância da ferramenta
api_tool = ApiMonitoringTool()

# --- AGENTES (RouteLLM) ---
data_collector_agent = Agent(
    role='Coletor de Métricas da API',
    goal='Coletar dados vitais dos endpoints de saúde.',
    backstory='Robô especializado em requisições HTTP para métricas.',
    tools=[api_tool],
    verbose=True,
    llm=llm,
    allow_delegation=False
)

data_analyzer_agent = Agent(
    role='Analista de Saúde da Aplicação',
    goal='Interpretar os dados coletados e detectar anomalias.',
    backstory='Especialista em identificação de falhas em sistemas.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

notification_agent = Agent(
    role='Gerador de Alertas',
    goal='Transformar os insights técnicos em alertas claros e objetivos.',
    backstory='Profissional em comunicação técnica para times de dev.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

# --- TAREFAS ---
collect_data_task = Task(
    description=(
        'Use a ferramenta "API Monitoring Tool" para buscar dados nos endpoints: '
        '`/actuator/health` e `/actuator/metrics/jvm.memory.used`. '
        'Combine as respostas em JSON unificado com chaves claras.'
    ),
    expected_output='JSON com dados brutos: {"health": {...}, "jvmMemoryUsed": {...}}',
    agent=data_collector_agent
)

analyze_data_task = Task(
    description=(
        'Analise os dados: '
        '- health.status deve ser "UP". '
        '- Se memória usada (em bytes) > 700MB (734003200 bytes) → levantar alerta. '
        'Produza um resumo curto e objetivo em português com conclusões e métricas.'
    ),
    expected_output='Relatório com status e uso de memória, incluindo se há alerta.',
    agent=data_analyzer_agent,
    context=[collect_data_task]
)

notify_task = Task(
    description=(
        'Com base na análise, redija a mensagem final de notificação. '
        'Se tudo ok → mensagem positiva com status e uso de memória. '
        'Se falha/alerta → mensagem de alerta clara, concisa e acionável.'
    ),
    expected_output='Mensagem final para equipe (1-3 parágrafos curtos).',
    agent=notification_agent,
    context=[analyze_data_task]
)

# --- CREW ---
api_monitoring_crew = Crew(
    agents=[data_collector_agent, data_analyzer_agent, notification_agent],
    tasks=[collect_data_task, analyze_data_task, notify_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    try:
        print("🚀 Iniciando monitoramento da API...")

        # Falha cedo se variáveis essenciais da ferramenta estiverem faltando
        missing = [k for k in ["API_BASE_URL", "MONITORING_API_KEY"] if not os.getenv(k)]
        if missing:
            raise RuntimeError(f"Variáveis ausentes: {', '.join(missing)}")

        result = api_monitoring_crew.kickoff()

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": str(result)
        }
        persist_data(log_entry)

        print("\n\n########################")
        print("## Resultado Final do Monitoramento:")
        print("########################\n")
        print(result)
        print("📁 Salvo em monitoring_logs.json")

    except Exception as e:
        print("❌ Erro durante execução:", str(e))
        persist_data({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        })