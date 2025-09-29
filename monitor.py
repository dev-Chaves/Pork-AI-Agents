import os
import json
import time
import math
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# =========================
# Carrega variáveis de ambiente
# =========================
load_dotenv()

# Variáveis essenciais
required_vars = ["OPENAI_API_KEY", "API_BASE_URL", "MONITORING_API_KEY"]
missing = [k for k in required_vars if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Variáveis ausentes no .env: {', '.join(missing)}")

# Thresholds com defaults razoáveis (você pode ajustar no .env)
MEMORY_ALERT_MB = int(os.getenv("MEMORY_ALERT_MB", "700"))
CPU_ALERT_PCT = float(os.getenv("CPU_ALERT_PCT", "85.0"))      # %
ERROR_RATE_SLO_PCT = float(os.getenv("ERROR_RATE_SLO_PCT", "1.0"))  # %
LATENCY_P95_SLO_MS = int(os.getenv("LATENCY_P95_SLO_MS", "800"))

# =========================
# Configuração RouteLLM (Abacus.AI)
# =========================
client = OpenAI(
    base_url="https://routellm.abacus.ai/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

class RouteLLMWrapper:
    """
    Wrapper para usar RouteLLM com CrewAI.
    Simula um objeto LLM compatível com CrewAI/LangChain.
    """
    def __init__(self, model="gpt-4o-mini", temperature=0):
        self.model = model
        self.temperature = temperature

    def __call__(self, messages, **kwargs):
        return self._generate(messages, **kwargs)

    def invoke(self, messages, **kwargs):
        return self._generate(messages, **kwargs)

    def _generate(self, messages, **kwargs):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", 1500),
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erro na chamada LLM: {str(e)}"

llm = RouteLLMWrapper(model="gpt-4o-mini", temperature=0)

# =========================
# Persistência de logs
# =========================
LOG_FILE = "monitoring_logs.json"

def persist_data(entry: dict):
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

# =========================
# Ferramenta de coleta com retry/backoff
# =========================
class ApiMonitoringTool(BaseTool):
    """
    Ferramenta para requisições aos endpoints de monitoramento da API.
    Usa API_BASE_URL e MONITORING_API_KEY do .env.
    """
    name: str = "API Monitoring Tool"
    description: str = "Faz GET em endpoints de monitoramento e retorna o texto da resposta."

    def _run(self, endpoint: str) -> str:
        return self._get_with_retry(endpoint)

    def _get_with_retry(self, endpoint: str, retries: int = 3, timeout: int = 15) -> str:
        api_base_url = os.getenv("API_BASE_URL")
        api_key = os.getenv("MONITORING_API_KEY")

        if not api_base_url or not api_key:
            return "Erro: Variáveis API_BASE_URL ou MONITORING_API_KEY não definidas."

        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": api_key
        }

        url = f"{api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        last_error = None
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                try:
                    return json.dumps(response.json(), ensure_ascii=False)
                except Exception:
                    return response.text
            except requests.exceptions.RequestException as e:
                last_error = e
                # backoff exponencial simples
                sleep_secs = 2 ** attempt
                time.sleep(sleep_secs)
        return f"Erro ao acessar {endpoint}: {last_error}"

api_tool = ApiMonitoringTool()

# =========================
# Agentes aprimorados
# =========================

# Agente 1: Coletor - agora coleta múltiplos endpoints e retorna JSON estrito
collector_prompt = f"""
Você é um coletor de métricas de uma aplicação Spring Boot (Actuator).
Use a ferramenta "API Monitoring Tool" para buscar e AGRUPAR os dados dos seguintes endpoints, retornando EXATAMENTE um JSON único, sem texto adicional:

- /actuator/health
- /actuator/info
- /actuator/loggers
- /actuator/httptrace (ou /actuator/httpexchanges, se o seu ambiente usar esse nome)
- /actuator/metrics/jvm.memory.used
- /actuator/metrics/jvm.memory.max
- /actuator/metrics/process.cpu.usage
- /actuator/metrics/http.server.requests

Regras:
- Caso algum endpoint não exista, coloque null no respectivo campo.
- Não escreva nenhum texto fora do JSON final.
- O JSON final deve ter este formato (exemplo de chaves; preencha com os dados reais obtidos):
{{
  "health": <obj ou null>,
  "info": <obj ou null>,
  "loggers": <obj ou null>,
  "httptrace": <obj ou null>,
  "metrics": {{
     "jvm.memory.used": <obj ou null>,
     "jvm.memory.max": <obj ou null>,
     "process.cpu.usage": <obj ou null>,
     "http.server.requests": <obj ou null>
  }}
}}
"""

data_collector_agent = Agent(
    role='Coletor de Métricas da API',
    goal='Coletar dados vitais dos endpoints de saúde, info, loggers, httptrace e métricas.',
    backstory='Especialista em requisições HTTP e normalização de respostas.',
    tools=[api_tool],
    verbose=True,
    llm=llm,
    allow_delegation=False
)

collect_data_task = Task(
    description=collector_prompt + '\nExecute as chamadas e retorne somente o JSON final.',
    expected_output='JSON único com as chaves: health, info, loggers, httptrace, metrics{...}',
    agent=data_collector_agent
)

# Agente 2: Analisador - aplica thresholds, calcula percentuais/MB e sugere ações
analyzer_prompt = f"""
Você é um analista de observabilidade. Receberá um JSON bruto com as respostas do Actuator e deve produzir um diagnóstico estruturado.
Siga as regras:

1) Converta quando possível:
   - jvm.memory.used.value (bytes) -> MB (arredonde para inteiro) e percentual de jvm.memory.max (se disponível).
   - process.cpu.usage (0..1) -> % (duas casas).
   - Em http.server.requests, se houver contadores por status, estime taxa de erro (5xx / total) em %.

2) Compare com thresholds (do ambiente):
   - MEMORY_ALERT_MB = {MEMORY_ALERT_MB} MB
   - CPU_ALERT_PCT = {CPU_ALERT_PCT} %
   - ERROR_RATE_SLO_PCT = {ERROR_RATE_SLO_PCT} %
   - LATENCY_P95_SLO_MS = {LATENCY_P95_SLO_MS} ms (se houver latências ou max disponíveis)

3) Produza saída EXATAMENTE neste formato JSON (sem texto fora do JSON):
{{
  "summary": "texto curto",
  "severity": "INFO|WARN|ALERT|CRITICAL",
  "findings": [
     {{
       "area": "HEALTH|MEMORY|CPU|HTTP|DB|LOGGERS|TRACE|RELEASE",
       "status": "OK|WARN|ALERT",
       "metric": "nome_da_metrica",
       "value": "valor_legível",
       "threshold": "limite_legível",
       "details": "explicação curta"
     }}
  ],
  "actions": [
     "ação recomendada 1",
     "ação recomendada 2"
  ],
  "numbers": {{
    "memory_used_mb": <int ou null>,
    "memory_max_mb": <int ou null>,
    "memory_used_pct": <float ou null>,
    "cpu_usage_pct": <float ou null>,
    "http_error_rate_pct": <float ou null>
  }}
}}

4) Priorize problemas reais (ex.: memória acima do limite, CPU alta, erro 5xx alto).
5) Se alguma métrica não existir, preencha campos numéricos com null e ajuste findings/status conforme.
"""

data_analyzer_agent = Agent(
    role='Analista de Saúde da Aplicação',
    goal='Interpretar dados coletados, detectar anomalias e recomendar ações.',
    backstory='Engenheiro de confiabilidade com foco em SLOs e performance.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

analyze_data_task = Task(
    description=analyzer_prompt + "\nEntrada: use o JSON retornado pelo coletor.",
    expected_output='JSON estruturado com summary, severity, findings, actions, numbers',
    agent=data_analyzer_agent,
    context=[collect_data_task]
)

# Agente 3: Notificador - mensagem curta e acionável
notifier_prompt = """
Você é um gerador de alertas. A partir do JSON de análise (summary, severity, findings, actions),
gere uma notificação curta, objetiva e acionável para a equipe.

Regras:
- Título curto com o status (ex.: [ALERT] Memória acima do limite).
- 1 a 2 parágrafos no máximo, em português, mencionando números-chave (memória/CPU/erros).
- Lista de 2 a 4 ações objetivas (bullets curtos).
- Sem verborragia. Texto final limpo (sem markdown pesado).
"""

notification_agent = Agent(
    role='Gerador de Alertas',
    goal='Converter a análise técnica em mensagem clara e acionável.',
    backstory='Especialista em comunicação técnica para times de engenharia.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

notify_task = Task(
    description=notifier_prompt + "\nEntrada: use o JSON estruturado do analisador.",
    expected_output='Mensagem final curta e acionável para a equipe.',
    agent=notification_agent,
    context=[analyze_data_task]
)

# =========================
# Orquestração
# =========================
api_monitoring_crew = Crew(
    agents=[data_collector_agent, data_analyzer_agent, notification_agent],
    tasks=[collect_data_task, analyze_data_task, notify_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    try:
        print("🚀 Iniciando monitoramento da API...")

        # Teste rápido da conexão com RouteLLM
        try:
            _ = llm("teste de conexão")
            print("✅ Conexão com RouteLLM OK")
        except Exception as e:
            print(f"⚠️ Aviso: Problema na conexão RouteLLM: {e}")

        result = api_monitoring_crew.kickoff()

        # Tenta salvar também os dados brutos coletados do primeiro task (se disponível no CrewAI)
        # Como fallback, salvamos apenas o resultado final.
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