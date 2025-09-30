import os
import json
import time
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
import requests
import schedule
from dotenv import load_dotenv
from openai import OpenAI

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# ====
# Carrega variáveis de ambiente
# ====
load_dotenv()

# Variáveis essenciais
required_vars = ["OPENAI_API_KEY", "API_BASE_URL", "MONITORING_API_KEY"]
missing = [k for k in required_vars if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Variáveis ausentes no .env: {', '.join(missing)}")

# Configurações do Telegram (opcionais)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Variável para controlar updates já processados
last_update_id = 0

# Thresholds com defaults razoáveis
MEMORY_ALERT_MB = int(os.getenv("MEMORY_ALERT_MB", "700"))
CPU_ALERT_PCT = float(os.getenv("CPU_ALERT_PCT", "85.0"))
ERROR_RATE_SLO_PCT = float(os.getenv("ERROR_RATE_SLO_PCT", "1.0"))
LATENCY_P95_SLO_MS = int(os.getenv("LATENCY_P95_SLO_MS", "800"))
LATENCY_AVG_SLO_MS = int(os.getenv("LATENCY_AVG_SLO_MS", "500"))

# ====
# Configuração RouteLLM (Abacus.AI)
# ====
client = OpenAI(
    base_url="https://routellm.abacus.ai/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

class RouteLLMWrapper:
    """Wrapper para usar RouteLLM com CrewAI."""
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
                max_tokens=kwargs.get("max_tokens", 2500),
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erro na chamada LLM: {str(e)}"

llm = RouteLLMWrapper(model="gpt-4o-mini", temperature=0)

# ====
# Persistência de logs e análises
# ====
LOG_FILE = "monitoring_logs.json"
ANALYTICS_FILE = "api_analytics.json"
EXCEPTIONS_FILE = "exceptions_report.json"

def persist_data(entry: dict, filename: str = LOG_FILE):
    """Persiste dados em arquivo JSON."""
    try:
        if not os.path.exists(filename):
            with open(filename, "w", encoding="utf-8") as f:
                json.dump([], f)
        with open(filename, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(entry)
            # Mantém apenas últimos 1000 registros
            if len(data) > 1000:
                data = data[-1000:]
            f.seek(0)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.truncate()
    except Exception as e:
        print(f"⚠️ Erro ao salvar em {filename}: {e}")

def load_data(filename: str) -> List[Dict]:
    """Carrega dados de arquivo JSON."""
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"⚠️ Erro ao carregar {filename}: {e}")
        return []

# ====
# Análise de Exceções e Estatísticas
# ====
class ExceptionAnalyzer:
    """Analisa exceções e gera estatísticas detalhadas."""
    
    @staticmethod
    def analyze_http_traces(traces_data: Any) -> Dict:
        """Analisa traces HTTP para extrair exceções e estatísticas."""
        analysis = {
            "total_requests": 0,
            "status_codes": Counter(),
            "exceptions": [],
            "exception_types": Counter(),
            "endpoints": Counter(),
            "methods": Counter(),
            "response_times": [],
            "errors_by_endpoint": defaultdict(list),
            "time_range": {"start": None, "end": None}
        }
        
        try:
            # Suporta diferentes formatos do Actuator
            exchanges = []
            if isinstance(traces_data, dict):
                exchanges = traces_data.get("exchanges", []) or traces_data.get("traces", [])
            elif isinstance(traces_data, list):
                exchanges = traces_data
            
            for exchange in exchanges:
                analysis["total_requests"] += 1
                
                # Extrai informações da requisição
                request = exchange.get("request", {})
                response = exchange.get("response", {})
                
                method = request.get("method", "UNKNOWN")
                uri = request.get("uri", "UNKNOWN")
                status = response.get("status", 0)
                timestamp = exchange.get("timestamp")
                
                # Tempo de resposta (se disponível)
                time_taken = exchange.get("timeTaken") or exchange.get("duration")
                if time_taken:
                    analysis["response_times"].append(time_taken)
                
                # Contadores
                analysis["status_codes"][status] += 1
                analysis["endpoints"][uri] += 1
                analysis["methods"][method] += 1
                
                # Detecta exceções (qualquer status >= 400)
                if status >= 400:
                    exception_info = {
                        "timestamp": timestamp,
                        "method": method,
                        "endpoint": uri,
                        "status": status,
                        "time_taken_ms": time_taken,
                        "error_type": ExceptionAnalyzer._classify_error(status),
                        "headers": response.get("headers", {}),
                        "session_id": exchange.get("sessionId")
                    }
                    
                    analysis["exceptions"].append(exception_info)
                    analysis["exception_types"][exception_info["error_type"]] += 1
                    analysis["errors_by_endpoint"][uri].append(exception_info)
                
                # Atualiza range de tempo
                if timestamp:
                    if not analysis["time_range"]["start"]:
                        analysis["time_range"]["start"] = timestamp
                    analysis["time_range"]["end"] = timestamp
            
        except Exception as e:
            print(f"⚠️ Erro ao analisar traces: {e}")
        
        return analysis
    
    @staticmethod
    def _classify_error(status: int) -> str:
        """Classifica tipo de erro baseado no status HTTP."""
        if status == 400:
            return "BAD_REQUEST"
        elif status == 401:
            return "UNAUTHORIZED"
        elif status == 403:
            return "FORBIDDEN"
        elif status == 404:
            return "NOT_FOUND"
        elif status == 408:
            return "REQUEST_TIMEOUT"
        elif status == 429:
            return "RATE_LIMIT"
        elif status == 500:
            return "INTERNAL_SERVER_ERROR"
        elif status == 502:
            return "BAD_GATEWAY"
        elif status == 503:
            return "SERVICE_UNAVAILABLE"
        elif status == 504:
            return "GATEWAY_TIMEOUT"
        elif 400 <= status < 500:
            return "CLIENT_ERROR"
        elif 500 <= status < 600:
            return "SERVER_ERROR"
        else:
            return "UNKNOWN_ERROR"
    
    @staticmethod
    def calculate_statistics(analysis: Dict) -> Dict:
        """Calcula estatísticas detalhadas."""
        stats = {
            "total_requests": analysis["total_requests"],
            "total_errors": len(analysis["exceptions"]),
            "error_rate_pct": 0.0,
            "status_distribution": dict(analysis["status_codes"]),
            "top_endpoints": analysis["endpoints"].most_common(10),
            "top_exception_types": dict(analysis["exception_types"]),
            "methods_distribution": dict(analysis["methods"]),
            "response_time_stats": {},
            "most_problematic_endpoints": []
        }
        
        # Taxa de erro
        if stats["total_requests"] > 0:
            stats["error_rate_pct"] = (stats["total_errors"] / stats["total_requests"]) * 100
        
        # Estatísticas de tempo de resposta
        if analysis["response_times"]:
            times = sorted(analysis["response_times"])
            stats["response_time_stats"] = {
                "min_ms": min(times),
                "max_ms": max(times),
                "avg_ms": statistics.mean(times),
                "median_ms": statistics.median(times),
                "p95_ms": times[int(len(times) * 0.95)] if len(times) > 0 else 0,
                "p99_ms": times[int(len(times) * 0.99)] if len(times) > 0 else 0,
            }
        
        # Endpoints mais problemáticos
        endpoint_errors = [
            {
                "endpoint": endpoint,
                "error_count": len(errors),
                "error_types": Counter([e["error_type"] for e in errors])
            }
            for endpoint, errors in analysis["errors_by_endpoint"].items()
        ]
        endpoint_errors.sort(key=lambda x: x["error_count"], reverse=True)
        stats["most_problematic_endpoints"] = endpoint_errors[:10]
        
        return stats

# ====
# Gerador de Relatórios
# ====
class ReportGenerator:
    """Gera relatórios detalhados de monitoramento."""
    
    @staticmethod
    def generate_comprehensive_report(analysis: Dict, stats: Dict, metrics: Dict) -> str:
        """Gera relatório completo em formato markdown."""
        report = []
        report.append("# 📊 RELATÓRIO DE MONITORAMENTO DA API\n")
        report.append(f"**Data/Hora:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        
        # Período analisado
        if analysis["time_range"]["start"]:
            report.append(f"**Período:** {analysis['time_range']['start']} até {analysis['time_range']['end']}\n")
        
        report.append("\n---\n")
        
        # Resumo Executivo
        report.append("## 🎯 RESUMO EXECUTIVO\n")
        report.append(f"- **Total de Requisições:** {stats['total_requests']:,}")
        report.append(f"- **Total de Erros:** {stats['total_errors']:,}")
        report.append(f"- **Taxa de Erro:** {stats['error_rate_pct']:.2f}%")
        
        if stats["response_time_stats"]:
            rt = stats["response_time_stats"]
            report.append(f"- **Tempo Médio de Resposta:** {rt['avg_ms']:.0f}ms")
            report.append(f"- **P95 Latência:** {rt['p95_ms']:.0f}ms")
            report.append(f"- **P99 Latência:** {rt['p99_ms']:.0f}ms")
        
        report.append("\n---\n")
        
        # Distribuição de Status HTTP
        report.append("## 📈 DISTRIBUIÇÃO DE STATUS HTTP\n")
        for status, count in sorted(stats["status_distribution"].items()):
            pct = (count / stats["total_requests"]) * 100 if stats["total_requests"] > 0 else 0
            emoji = "✅" if status < 400 else "⚠️" if status < 500 else "❌"
            report.append(f"{emoji} **{status}:** {count:,} ({pct:.1f}%)")
        
        report.append("\n---\n")
        
        # Tipos de Exceções
        if stats["top_exception_types"]:
            report.append("## 🚨 TIPOS DE EXCEÇÕES MAIS FREQUENTES\n")
            for exc_type, count in sorted(stats["top_exception_types"].items(), 
                                         key=lambda x: x[1], reverse=True):
                pct = (count / stats["total_errors"]) * 100 if stats["total_errors"] > 0 else 0
                report.append(f"- **{exc_type}:** {count:,} ({pct:.1f}%)")
        
        report.append("\n---\n")
        
        # Endpoints Mais Usados
        report.append("## 🔥 TOP 10 ENDPOINTS MAIS REQUISITADOS\n")
        for endpoint, count in stats["top_endpoints"]:
            pct = (count / stats["total_requests"]) * 100 if stats["total_requests"] > 0 else 0
            report.append(f"- `{endpoint}`: {count:,} ({pct:.1f}%)")
        
        report.append("\n---\n")
        
        # Endpoints Problemáticos
        if stats["most_problematic_endpoints"]:
            report.append("## ⚠️ ENDPOINTS MAIS PROBLEMÁTICOS\n")
            for item in stats["most_problematic_endpoints"][:5]:
                report.append(f"\n### `{item['endpoint']}`")
                report.append(f"- **Total de Erros:** {item['error_count']}")
                report.append("- **Tipos de Erro:**")
                for err_type, count in item["error_types"].items():
                    report.append(f"  - {err_type}: {count}")
        
        report.append("\n---\n")
        
        # Métricas de Sistema
        report.append("## 💻 MÉTRICAS DE SISTEMA\n")
        if metrics.get("memory_used_mb"):
            report.append(f"- **Memória Usada:** {metrics['memory_used_mb']}MB / {metrics.get('memory_max_mb', 'N/A')}MB ({metrics.get('memory_used_pct', 0):.1f}%)")
        if metrics.get("cpu_usage_pct"):
            report.append(f"- **CPU:** {metrics['cpu_usage_pct']:.1f}%")
        
        report.append("\n---\n")
        
        # Tempos de Resposta Detalhados
        if stats["response_time_stats"]:
            report.append("## ⚡ ANÁLISE DE PERFORMANCE\n")
            rt = stats["response_time_stats"]
            report.append(f"- **Mínimo:** {rt['min_ms']:.0f}ms")
            report.append(f"- **Máximo:** {rt['max_ms']:.0f}ms")
            report.append(f"- **Média:** {rt['avg_ms']:.0f}ms")
            report.append(f"- **Mediana:** {rt['median_ms']:.0f}ms")
            report.append(f"- **P95:** {rt['p95_ms']:.0f}ms")
            report.append(f"- **P99:** {rt['p99_ms']:.0f}ms")
            
            # Alertas de performance
            if rt['avg_ms'] > LATENCY_AVG_SLO_MS:
                report.append(f"\n⚠️ **ALERTA:** Tempo médio ({rt['avg_ms']:.0f}ms) acima do SLO ({LATENCY_AVG_SLO_MS}ms)")
            if rt['p95_ms'] > LATENCY_P95_SLO_MS:
                report.append(f"\n⚠️ **ALERTA:** P95 ({rt['p95_ms']:.0f}ms) acima do SLO ({LATENCY_P95_SLO_MS}ms)")
        
        report.append("\n---\n")
        
        # Distribuição de Métodos HTTP
        report.append("## 🔄 DISTRIBUIÇÃO DE MÉTODOS HTTP\n")
        for method, count in sorted(stats["methods_distribution"].items(), 
                                    key=lambda x: x[1], reverse=True):
            pct = (count / stats["total_requests"]) * 100 if stats["total_requests"] > 0 else 0
            report.append(f"- **{method}:** {count:,} ({pct:.1f}%)")
        
        return "\n".join(report)

# ====
# Notificação via Telegram
# ====
def send_telegram_notification(message: str, parse_mode: str = None):
    """Envia notificação via Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram não configurado. Pulando notificação.")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        
        # Divide mensagens longas
        max_length = 4000
        if len(message) > max_length:
            parts = [message[i:i+max_length] for i in range(0, len(message), max_length)]
            for part in parts:
                data = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": part,
                    "disable_web_page_preview": True
                }
                if parse_mode:
                    data["parse_mode"] = parse_mode
                
                response = requests.post(url, json=data, timeout=10)
                response.raise_for_status()
                time.sleep(1)  # Evita rate limit
        else:
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "disable_web_page_preview": True
            }
            if parse_mode:
                data["parse_mode"] = parse_mode
            
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
        
        print("✅ Notificação enviada via Telegram")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao enviar Telegram: {e}")
        return False

# ====
# Escuta de comandos do Telegram
# ====
def listen_for_commands():
    """Faz polling no Telegram para ouvir comandos."""
    global last_update_id
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {"offset": last_update_id + 1, "timeout": 1}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if "result" in data and data["result"]:
            for update in data["result"]:
                last_update_id = update["update_id"]
                
                if "message" in update:
                    chat_id = update["message"]["chat"]["id"]
                    text = update["message"].get("text", "").strip()
                    username = update["message"]["from"].get("username", "Usuário")

                    if str(chat_id) != str(TELEGRAM_CHAT_ID):
                        continue

                    print(f"📱 Comando recebido de @{username}: {text}")

                    if text == "/status":
                        send_telegram_notification("✅ Bot está rodando normalmente!")
                    
                    elif text == "/run":
                        send_telegram_notification("🚀 Executando monitoramento sob comando...")
                        run_monitoring()
                    
                    elif text == "/report":
                        send_telegram_notification("📊 Gerando relatório completo...")
                        generate_and_send_report()
                    
                    elif text == "/exceptions":
                        send_telegram_notification("🚨 Gerando relatório de exceções...")
                        generate_exceptions_report()
                    
                    elif text == "/stats":
                        send_telegram_notification("📈 Gerando estatísticas...")
                        generate_stats_report()
                    
                    elif text == "/help":
                        help_msg = (
                            "📖 Comandos disponíveis:\n\n"
                            "/status - Ver se o bot está online\n"
                            "/run - Executar monitoramento agora\n"
                            "/report - Relatório completo\n"
                            "/exceptions - Relatório de exceções\n"
                            "/stats - Estatísticas detalhadas\n"
                            "/logs - Ver últimos logs\n"
                            "/config - Ver configurações\n"
                            "/help - Mostrar esta ajuda"
                        )
                        send_telegram_notification(help_msg)
                    
                    elif text == "/logs":
                        logs = load_data(LOG_FILE)
                        if logs:
                            last_log = logs[-1]
                            timestamp = last_log.get("timestamp", "N/A")
                            log_msg = f"📋 Último log:\n\n⏰ {timestamp}\n\n"
                            if "error" in last_log:
                                log_msg += f"❌ Erro: {last_log['error']}"
                            else:
                                log_msg += "✅ Execução bem-sucedida"
                            send_telegram_notification(log_msg)
                        else:
                            send_telegram_notification("📋 Nenhum log encontrado.")
                    
                    elif text == "/config":
                        config_msg = (
                            f"⚙️ Configurações atuais:\n\n"
                            f"• Memória Alert: {MEMORY_ALERT_MB}MB\n"
                            f"• CPU Alert: {CPU_ALERT_PCT}%\n"
                            f"• Error Rate SLO: {ERROR_RATE_SLO_PCT}%\n"
                            f"• Latency AVG SLO: {LATENCY_AVG_SLO_MS}ms\n"
                            f"• Latency P95 SLO: {LATENCY_P95_SLO_MS}ms\n"
                            f"• Telegram: {'✅ Configurado' if TELEGRAM_BOT_TOKEN else '❌ Não configurado'}"
                        )
                        send_telegram_notification(config_msg)
                    
                    else:
                        send_telegram_notification(
                            f"❓ Comando não reconhecido: {text}\n\n"
                            "Digite /help para ver os comandos disponíveis."
                        )

    except Exception as e:
        print(f"⚠️ Erro ao ouvir comandos: {e}")

# ====
# Ferramenta de coleta com retry/backoff
# ====
class ApiMonitoringTool(BaseTool):
    """Ferramenta para requisições aos endpoints de monitoramento."""
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
                sleep_secs = 2 ** attempt
                time.sleep(sleep_secs)
        return f"Erro ao acessar {endpoint}: {last_error}"

api_tool = ApiMonitoringTool()

# ====
# Agentes Aprimorados
# ====

# Agente 1: Coletor Avançado
collector_prompt = """
Você é um coletor avançado de métricas de uma aplicação Spring Boot (Actuator).
Use a ferramenta "API Monitoring Tool" para buscar TODOS os dados dos seguintes endpoints:

ENDPOINTS OBRIGATÓRIOS:
- /actuator/health
- /actuator/info
- /actuator/metrics/jvm.memory.used
- /actuator/metrics/jvm.memory.max
- /actuator/metrics/process.cpu.usage
- /actuator/metrics/http.server.requests

ENDPOINTS DE TRACES (tente ambos, use o que funcionar):
- /actuator/httptrace
- /actuator/httpexchanges

ENDPOINTS ADICIONAIS (se disponíveis):
- /actuator/loggers
- /actuator/metrics/jvm.threads.live
- /actuator/metrics/jvm.gc.pause
- /actuator/metrics/system.cpu.usage

Retorne EXATAMENTE um JSON único com TODOS os dados coletados:
{
  "health": <obj ou null>,
  "info": <obj ou null>,
  "httptrace": <obj ou null>,
  "httpexchanges": <obj ou null>,
  "loggers": <obj ou null>,
  "metrics": {
    "jvm.memory.used": <obj ou null>,
    "jvm.memory.max": <obj ou null>,
    "process.cpu.usage": <obj ou null>,
    "http.server.requests": <obj ou null>,
    "jvm.threads.live": <obj ou null>,
    "jvm.gc.pause": <obj ou null>,
    "system.cpu.usage": <obj ou null>
  }
}

IMPORTANTE: Não adicione texto fora do JSON. Se um endpoint falhar, coloque null.
"""

data_collector_agent = Agent(
    role='Coletor Avançado de Métricas',
    goal='Coletar TODOS os dados disponíveis dos endpoints de monitoramento.',
    backstory='Especialista em coleta exaustiva de métricas e traces de aplicações.',
    tools=[api_tool],
    verbose=True,
    llm=llm,
    allow_delegation=False
)

collect_data_task = Task(
    description=collector_prompt,
    expected_output='JSON completo com todos os dados coletados',
    agent=data_collector_agent
)

# Agente 2: Analisador de Exceções
exception_analyzer_prompt = f"""
Você é um especialista em análise de exceções e erros de APIs.
Receberá um JSON com dados do Actuator e deve fazer uma análise PROFUNDA de TODAS as exceções.

ANÁLISE OBRIGATÓRIA:
1. TODAS as exceções (não só 500, mas 400, 401, 403, 404, 408, 429, 502, 503, 504, etc)
2. Frequência de cada tipo de exceção
3. Endpoints que mais geram erros
4. Padrões temporais (se houver timestamps)
5. Correlação entre tipos de erro e endpoints
6. Taxa de erro geral e por endpoint
7. Classificação de severidade

MÉTRICAS DE SISTEMA:
- Converter memória para MB e percentual
- CPU em percentual
- Threads ativas
- GC pause time (se disponível)

THRESHOLDS:
- MEMORY_ALERT_MB = {MEMORY_ALERT_MB}
- CPU_ALERT_PCT = {CPU_ALERT_PCT}
- ERROR_RATE_SLO_PCT = {ERROR_RATE_SLO_PCT}
- LATENCY_AVG_SLO_MS = {LATENCY_AVG_SLO_MS}
- LATENCY_P95_SLO_MS = {LATENCY_P95_SLO_MS}

Retorne EXATAMENTE este JSON:
{{
  "summary": "resumo executivo",
  "severity": "INFO|WARN|ALERT|CRITICAL",
  "exceptions_analysis": {{
    "total_exceptions": <int>,
    "exception_types": {{"tipo": count}},
    "exceptions_by_status": {{"status": count}},
    "most_problematic_endpoints": [
      {{"endpoint": "uri", "error_count": <int>, "error_types": {{}}}
    ],
    "error_rate_pct": <float>,
    "client_errors_4xx": <int>,
    "server_errors_5xx": <int>
  }},
  "performance_analysis": {{
    "total_requests": <int>,
    "avg_response_time_ms": <float ou null>,
    "p95_response_time_ms": <float ou null>,
    "p99_response_time_ms": <float ou null>,
    "slowest_endpoints": []
  }},
  "system_metrics": {{
    "memory_used_mb": <int ou null>,
    "memory_max_mb": <int ou null>,
    "memory_used_pct": <float ou null>,
    "cpu_usage_pct": <float ou null>,
    "threads_live": <int ou null>
  }},
  "findings": [
    {{
      "area": "EXCEPTIONS|PERFORMANCE|MEMORY|CPU",
      "status": "OK|WARN|ALERT|CRITICAL",
      "metric": "nome",
      "value": "valor",
      "threshold": "limite",
      "details": "explicação"
    }}
  ],
  "actions": ["ação 1", "ação 2"]
}}
"""

exception_analyzer_agent = Agent(
    role='Analista de Exceções e Performance',
    goal='Analisar TODAS as exceções, erros e métricas de performance em profundidade.',
    backstory='Engenheiro de confiabilidade especializado em análise de falhas e otimização.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

analyze_exceptions_task = Task(
    description=exception_analyzer_prompt,
    expected_output='JSON estruturado com análise completa de exceções e performance',
    agent=exception_analyzer_agent,
    context=[collect_data_task]
)

# Agente 3: Gerador de Relatório
report_generator_prompt = """
Você é um gerador de relatórios executivos para equipes de engenharia.
Receberá uma análise completa e deve gerar um relatório CONCISO e ACIONÁVEL.

ESTRUTURA DO RELATÓRIO:
1. Status Geral (emoji + resumo de 1 linha)
2. Métricas Principais (3-5 números-chave)
3. Problemas Críticos (se houver)
4. Top 3 Tipos de Exceções
5. Top 3 Endpoints Problemáticos
6. Ações Recomendadas (máximo 5, priorizadas)

REGRAS:
- Máximo 15 linhas
- Use emojis para facilitar leitura
- Números formatados (ex: 1,234 ou 45.2%)
- Priorize informações acionáveis
- Destaque alertas críticos

Formato de saída: texto limpo, sem markdown pesado.
"""

report_generator_agent = Agent(
    role='Gerador de Relatórios Executivos',
    goal='Criar relatórios concisos e acionáveis para a equipe.',
    backstory='Especialista em comunicação técnica e priorização de problemas.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

generate_report_task = Task(
    description=report_generator_prompt,
    expected_output='Relatório executivo conciso e acionável',
    agent=report_generator_agent,
    context=[analyze_exceptions_task]
)

# ====
# Orquestração
# ====
api_monitoring_crew = Crew(
    agents=[data_collector_agent, exception_analyzer_agent, report_generator_agent],
    tasks=[collect_data_task, analyze_exceptions_task, generate_report_task],
    process=Process.sequential,
    verbose=True
)

# ====
# Funções de Relatório
# ====
def generate_and_send_report():
    """Gera e envia relatório completo."""
    try:
        analytics = load_data(ANALYTICS_FILE)
        if not analytics:
            send_telegram_notification("⚠️ Nenhum dado de análise disponível ainda.")
            return
        
        latest = analytics[-1]
        analysis = latest.get("analysis", {})
        stats = latest.get("statistics", {})
        metrics = latest.get("system_metrics", {})
        
        report = ReportGenerator.generate_comprehensive_report(analysis, stats, metrics)
        
        # Salva relatório em arquivo
        report_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        send_telegram_notification(f"📊 Relatório completo gerado: {report_file}\n\n{report[:3000]}...")
        print(f"✅ Relatório salvo em {report_file}")
        
    except Exception as e:
        send_telegram_notification(f"❌ Erro ao gerar relatório: {e}")

def generate_exceptions_report():
    """Gera relatório focado em exceções."""
    try:
        exceptions_data = load_data(EXCEPTIONS_FILE)
        if not exceptions_data:
            send_telegram_notification("⚠️ Nenhum dado de exceções disponível.")
            return
        
        latest = exceptions_data[-1]
        
        report = []
        report.append("🚨 RELATÓRIO DE EXCEÇÕES\n")
        report.append(f"Total de Exceções: {latest.get('total_exceptions', 0)}")
        report.append(f"Taxa de Erro: {latest.get('error_rate_pct', 0):.2f}%\n")
        
        report.append("Top Tipos de Exceção:")
        for exc_type, count in list(latest.get('exception_types', {}).items())[:5]:
            report.append(f"  • {exc_type}: {count}")
        
        report.append("\nEndpoints Mais Problemáticos:")
        for ep in latest.get('most_problematic_endpoints', [])[:5]:
            report.append(f"  • {ep['endpoint']}: {ep['error_count']} erros")
        
        send_telegram_notification("\n".join(report))
        
    except Exception as e:
        send_telegram_notification(f"❌ Erro ao gerar relatório de exceções: {e}")

def generate_stats_report():
    """Gera relatório de estatísticas."""
    try:
        analytics = load_data(ANALYTICS_FILE)
        if not analytics:
            send_telegram_notification("⚠️ Nenhum dado estatístico disponível.")
            return
        
        latest = analytics[-1]
        stats = latest.get("statistics", {})
        
        report = []
        report.append("📈 ESTATÍSTICAS DA API\n")
        report.append(f"Total de Requisições: {stats.get('total_requests', 0):,}")
        report.append(f"Total de Erros: {stats.get('total_errors', 0):,}")
        report.append(f"Taxa de Erro: {stats.get('error_rate_pct', 0):.2f}%\n")
        
        if stats.get("response_time_stats"):
            rt = stats["response_time_stats"]
            report.append("Tempos de Resposta:")
            report.append(f"  • Média: {rt.get('avg_ms', 0):.0f}ms")
            report.append(f"  • P95: {rt.get('p95_ms', 0):.0f}ms")
            report.append(f"  • P99: {rt.get('p99_ms', 0):.0f}ms\n")
        
        report.append("Top Endpoints:")
        for endpoint, count in stats.get("top_endpoints", [])[:5]:
            report.append(f"  • {endpoint}: {count:,}")
        
        send_telegram_notification("\n".join(report))
        
    except Exception as e:
        send_telegram_notification(f"❌ Erro ao gerar estatísticas: {e}")

# ====
# Função principal de monitoramento
# ====
def run_monitoring():
    """Executa o monitoramento completo."""
    start_time = time.time()
    
    try:
        print("🚀 Iniciando monitoramento avançado da API...")

        # Executa o monitoramento
        result = api_monitoring_crew.kickoff()
        
        response_time = time.time() - start_time
        
        # Extrai dados de análise
        analysis_data = None
        collected_data = None
        
        try:
            if hasattr(result, 'tasks_output') and len(result.tasks_output) >= 2:
                # Dados coletados
                collected_output = str(result.tasks_output[0])
                collected_data = json.loads(collected_output)
                
                # Análise
                analysis_output = str(result.tasks_output[1])
                analysis_data = json.loads(analysis_output)
                
                # Processa traces para análise detalhada
                traces = collected_data.get("httptrace") or collected_data.get("httpexchanges")
                if traces:
                    analyzer = ExceptionAnalyzer()
                    trace_analysis = analyzer.analyze_http_traces(traces)
                    statistics_data = analyzer.calculate_statistics(trace_analysis)
                    
                    # Salva análise detalhada
                    analytics_entry = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "analysis": trace_analysis,
                        "statistics": statistics_data,
                        "system_metrics": analysis_data.get("system_metrics", {})
                    }
                    persist_data(analytics_entry, ANALYTICS_FILE)
                    
                    # Salva exceções
                    exceptions_entry = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "total_exceptions": len(trace_analysis["exceptions"]),
                        "exception_types": dict(trace_analysis["exception_types"]),
                        "error_rate_pct": statistics_data.get("error_rate_pct", 0),
                        "most_problematic_endpoints": statistics_data.get("most_problematic_endpoints", [])
                    }
                    persist_data(exceptions_entry, EXCEPTIONS_FILE)
                    
                    print(f"✅ Análise detalhada salva:")
                    print(f"   - Total de requisições: {statistics_data['total_requests']}")
                    print(f"   - Total de exceções: {statistics_data['total_errors']}")
                    print(f"   - Taxa de erro: {statistics_data['error_rate_pct']:.2f}%")
                    
        except Exception as e:
            print(f"⚠️ Erro ao processar análise detalhada: {e}")

        # Salva log
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response_time_seconds": round(response_time, 2),
            "result": str(result),
            "analysis_data": analysis_data
        }
        persist_data(log_entry, LOG_FILE)

        print(f"\n####")
        print(f"## Resultado Final do Monitoramento:")
        print(f"## Tempo de resposta: {response_time:.2f}s")
        print(f"####\n")
        print(result)

        # Envia notificação
        notification_message = f"⚡ Tempo: {response_time:.1f}s\n\n{str(result)}"
        send_telegram_notification(notification_message)

    except Exception as e:
        error_msg = f"❌ Erro durante execução: {str(e)}"
        print(error_msg)
        
        persist_data({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "response_time_seconds": round(time.time() - start_time, 2)
        }, LOG_FILE)
        
        send_telegram_notification(f"🚨 ERRO no Monitor\n\n{error_msg}")

# ====
# Agendamento e execução
# ====
if __name__ == "__main__":
    print("🤖 Monitor Avançado de API iniciado!")
    print("📅 Agendado para rodar diariamente às 12:00")
    print("🔍 Monitoramento de TODAS as exceções ativado")
    
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        print("✅ Telegram configurado")
        print("💬 Comandos: /help, /run, /report, /exceptions, /stats")
    else:
        print("⚠️ Telegram não configurado")
    
    # Agenda execução
    schedule.every().day.at("12:00").do(run_monitoring)
    
    # Execução inicial
    print("\n🧪 Executando teste inicial...")
    run_monitoring()
    
    print(f"\n⏰ Aguardando próxima execução...")
    print("💡 Pressione Ctrl+C para parar")
    
    # Loop principal
    try:
        while True:
            schedule.run_pending()
            listen_for_commands()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n👋 Monitor interrompido")