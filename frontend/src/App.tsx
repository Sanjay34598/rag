import { useState, useEffect } from 'react'
import { Mic, Activity, CheckCircle2, Server, Database, ShieldCheck, Cpu } from 'lucide-react'

interface HealthStatus {
  status: string
  service: string
  version: string
}

export default function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/health')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((data) => {
        setHealth(data)
        setLoading(false)
      })
      .catch((err) => {
        fetch('http://127.0.0.1:8000/health')
          .then((res) => res.json())
          .then((data) => {
            setHealth(data)
            setLoading(false)
          })
          .catch(() => {
            setError(err.message || 'Failed to connect to backend API')
            setLoading(false)
          })
      })
  }, [])

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6">
      <div className="max-w-3xl w-full bg-slate-900/80 border border-slate-800 rounded-2xl p-8 shadow-2xl backdrop-blur-md">
        
        {/* Header */}
        <div className="flex items-center gap-4 mb-6 pb-6 border-b border-slate-800">
          <div className="w-12 h-12 rounded-xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
            <Mic className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
              Voice-Enabled RAG System
            </h1>
            <p className="text-sm text-slate-400">HH Goa 2026 Shortlisting Task 2</p>
          </div>
        </div>

        {/* System Architecture Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 flex items-start gap-3">
            <Server className="w-5 h-5 text-indigo-400 mt-0.5 shrink-0" />
            <div>
              <h3 className="text-sm font-semibold text-slate-200">FastAPI Backend Skeleton</h3>
              <p className="text-xs text-slate-400 mt-1">Modular architecture initialized (`/app/api`, `/services`, `/models`)</p>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 flex items-start gap-3">
            <Database className="w-5 h-5 text-cyan-400 mt-0.5 shrink-0" />
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Retrieval Pipeline Ready</h3>
              <p className="text-xs text-slate-400 mt-1">Prepared for MSMARCO-XI, FAISS & BM25 hybrid search</p>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 flex items-start gap-3">
            <Mic className="w-5 h-5 text-emerald-400 mt-0.5 shrink-0" />
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Sarvam STT Engine</h3>
              <p className="text-xs text-slate-400 mt-1">Speech-to-text integration framework setup</p>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 flex items-start gap-3">
            <ShieldCheck className="w-5 h-5 text-purple-400 mt-0.5 shrink-0" />
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Guardrails & Latency Harness</h3>
              <p className="text-xs text-slate-400 mt-1">Safety checks & P50/P70/P100 latency measurement target &lt;200ms</p>
            </div>
          </div>
        </div>

        {/* Backend Connection Status */}
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-5 h-5 text-slate-400" />
            <span className="text-sm font-medium text-slate-300">Backend API Health:</span>
          </div>

          {loading ? (
            <div className="flex items-center gap-2 text-amber-400 text-xs font-mono">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping"></span>
              Checking endpoint...
            </div>
          ) : health ? (
            <div className="flex items-center gap-2 text-emerald-400 text-xs font-mono bg-emerald-500/10 px-3 py-1.5 rounded-lg border border-emerald-500/20">
              <CheckCircle2 className="w-4 h-4" />
              <span>{health.status.toUpperCase()}</span>
              <span className="text-slate-500">|</span>
              <span className="text-slate-400">{health.service} v{health.version}</span>
            </div>
          ) : (
            <div className="text-rose-400 text-xs font-mono bg-rose-500/10 px-3 py-1.5 rounded-lg border border-rose-500/20">
              Offline ({error || 'Backend offline'})
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-xs text-slate-500 flex items-center justify-center gap-2">
          <Cpu className="w-3.5 h-3.5" />
          <span>Stage 1 Foundation complete. Waiting for user instruction.</span>
        </div>

      </div>
    </div>
  )
}
