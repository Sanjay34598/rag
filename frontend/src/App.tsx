import { useState, useEffect, useRef } from 'react'
import { 
  Mic, Square, Loader2, CheckCircle2, AlertCircle, 
  Search, FileText, Cpu, Clock, ShieldCheck, Database, Volume2
} from 'lucide-react'

interface LatencyBreakdown {
  stt_ms: number
  retrieval_ms: number
  context_ms: number
  llm_ms: number
  grounding_ms: number
  total_ms: number
}

interface SourceChunk {
  chunk_id: string
  score: number
  text: string
}

interface VoiceQueryResponse {
  transcript: string
  answer: string
  grounded: boolean
  confidence: number
  sources: SourceChunk[]
  latency: LatencyBreakdown
}

type UIState = 'IDLE' | 'RECORDING' | 'PROCESSING' | 'SUCCESS' | 'ERROR'

export default function App() {
  const [uiState, setUiState] = useState<UIState>('IDLE')
  const [recordingTime, setRecordingTime] = useState<number>(0)
  const [textQuery, setTextQuery] = useState<string>('')
  const [result, setResult] = useState<VoiceQueryResponse | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  const startRecording = async () => {
    setErrorMsg(null)
    setResult(null)
    audioChunksRef.current = []

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        // Stop stream tracks
        stream.getTracks().forEach((track) => track.stop())

        const audioBlob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType || 'audio/webm' })
        await submitVoiceQuery(audioBlob, mediaRecorder.mimeType)
      }

      mediaRecorder.start()
      setUiState('RECORDING')
      setRecordingTime(0)

      timerRef.current = window.setInterval(() => {
        setRecordingTime((prev) => prev + 1)
      }, 1000)
    } catch (err: any) {
      console.error('Microphone permission or recording error:', err)
      setErrorMsg(err.message || 'Microphone access denied or not supported by browser.')
      setUiState('ERROR')
    }
  }

  const stopRecording = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
      setUiState('PROCESSING')
    }
  }

  const submitVoiceQuery = async (audioBlob: Blob, mimeType: string) => {
    const formData = new FormData()
    const filename = mimeType.includes('webm') ? 'recording.webm' : 'recording.wav'
    formData.append('file', audioBlob, filename)

    try {
      const endpoints = ['/api/v1/voice/query', 'http://127.0.0.1:8000/api/v1/voice/query']
      let response: Response | null = null
      let fetchError: Error | null = null

      for (const url of endpoints) {
        try {
          const res = await fetch(url, {
            method: 'POST',
            body: formData,
          })
          if (res.ok || res.status === 400 || res.status === 422 || res.status === 503 || res.status === 504) {
            response = res
            break
          }
        } catch (e: any) {
          fetchError = e
        }
      }

      if (!response) {
        throw fetchError || new Error('Failed to connect to Voice RAG API endpoint.')
      }

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Voice processing failed.')
      }

      setResult(data)
      setUiState('SUCCESS')
    } catch (err: any) {
      console.error('Voice Query API Error:', err)
      setErrorMsg(err.message || 'Error communicating with Voice RAG server.')
      setUiState('ERROR')
    }
  }

  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!textQuery.trim()) return

    setUiState('PROCESSING')
    setErrorMsg(null)
    setResult(null)

    try {
      const endpoints = ['/api/v1/rag/answer', 'http://127.0.0.1:8000/api/v1/rag/answer']
      let response: Response | null = null

      for (const url of endpoints) {
        try {
          const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: textQuery }),
          })
          if (res.ok) {
            response = res
            break
          }
        } catch (e) {
          // try next
        }
      }

      if (!response || !response.ok) {
        throw new Error('Failed to fetch answer for text query.')
      }

      const data = await response.json()
      setResult({
        transcript: textQuery,
        answer: data.answer,
        grounded: data.grounded,
        confidence: data.confidence,
        sources: data.sources,
        latency: {
          stt_ms: 0,
          retrieval_ms: data.latency.retrieval_ms,
          context_ms: data.latency.context_ms,
          llm_ms: data.latency.llm_ms,
          grounding_ms: data.latency.grounding_ms,
          total_ms: data.latency.total_ms,
        },
      })
      setUiState('SUCCESS')
    } catch (err: any) {
      setErrorMsg(err.message || 'Text query processing failed.')
      setUiState('ERROR')
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-start p-4 sm:p-6 font-sans">
      <div className="max-w-4xl w-full bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-xl">
        
        {/* Header */}
        <div className="flex items-center justify-between pb-6 mb-8 border-b border-slate-800">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-2xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
              <Volume2 className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-r from-indigo-400 via-cyan-400 to-emerald-400 bg-clip-text text-transparent">
                Voice RAG System (Hindi MSMARCO-XI)
              </h1>
              <p className="text-xs text-slate-400 mt-0.5">Sarvam Speech-to-Text + FAISS / BM25 Hybrid Retrieval</p>
            </div>
          </div>
          <div className="hidden sm:flex items-center gap-2 bg-slate-950/80 px-3 py-1.5 rounded-full border border-slate-800 text-xs text-slate-400">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Target &lt;200ms Retrieval</span>
          </div>
        </div>

        {/* Voice Recorder Control Section */}
        <div className="flex flex-col items-center justify-center p-8 bg-slate-950/70 border border-slate-800/80 rounded-2xl mb-8 relative overflow-hidden">
          
          {/* Status Glow Indicator */}
          {uiState === 'RECORDING' && (
            <div className="absolute inset-0 bg-rose-600/5 animate-pulse pointer-events-none" />
          )}

          {/* Microphone Main Action Button */}
          {uiState === 'RECORDING' ? (
            <button
              onClick={stopRecording}
              className="w-24 h-24 rounded-full bg-rose-600 hover:bg-rose-500 text-white flex flex-col items-center justify-center gap-1 shadow-lg shadow-rose-600/30 transition-all transform hover:scale-105 active:scale-95 border-4 border-rose-400/30"
              title="Click to stop and search"
            >
              <Square className="w-8 h-8 fill-current" />
              <span className="text-[10px] font-bold uppercase tracking-wider">Stop</span>
            </button>
          ) : uiState === 'PROCESSING' ? (
            <div className="w-24 h-24 rounded-full bg-indigo-600/20 border border-indigo-500/40 text-indigo-400 flex flex-col items-center justify-center gap-2">
              <Loader2 className="w-8 h-8 animate-spin" />
            </div>
          ) : (
            <button
              onClick={startRecording}
              className="w-24 h-24 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white flex flex-col items-center justify-center gap-1 shadow-lg shadow-indigo-600/30 transition-all transform hover:scale-105 active:scale-95 border-4 border-indigo-400/30"
              title="Click microphone to ask in Hindi"
            >
              <Mic className="w-9 h-9" />
              <span className="text-[10px] font-bold uppercase tracking-wider">Speak</span>
            </button>
          )}

          {/* Status Messages */}
          <div className="mt-4 text-center">
            {uiState === 'IDLE' && (
              <p className="text-sm text-slate-300 font-medium">माइक्रोफ़ोन पर क्लिक करें और हिंदी में प्रश्न पूछें</p>
            )}
            {uiState === 'RECORDING' && (
              <div className="flex items-center gap-2 text-rose-400 font-mono text-sm font-semibold">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-ping" />
                <span>Listening... ({formatTime(recordingTime)}) - Click to stop</span>
              </div>
            )}
            {uiState === 'PROCESSING' && (
              <p className="text-sm text-indigo-400 font-medium animate-pulse">
                Transcribing audio with Sarvam STT & searching vector index...
              </p>
            )}
            {uiState === 'SUCCESS' && (
              <p className="text-sm text-emerald-400 font-medium">Query processed successfully</p>
            )}
            {uiState === 'ERROR' && (
              <p className="text-sm text-rose-400 font-medium">Error processing voice input</p>
            )}
          </div>

          {/* Text Query Input Option */}
          <form onSubmit={handleTextSubmit} className="mt-6 w-full max-w-lg flex items-center gap-2">
            <input
              type="text"
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              placeholder="या हिंदी में टाइप करें (Or type text query)..."
              className="flex-1 bg-slate-900 border border-slate-700/80 rounded-xl px-4 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
            />
            <button
              type="submit"
              disabled={uiState === 'PROCESSING' || !textQuery.trim()}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white px-4 py-2.5 rounded-xl text-sm font-medium transition flex items-center gap-2"
            >
              <Search className="w-4 h-4" />
              <span>Search</span>
            </button>
          </form>
        </div>

        {/* Error Alert Box */}
        {errorMsg && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-start gap-3 mb-8">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <h4 className="text-sm font-semibold text-rose-300">Processing Error</h4>
              <p className="text-xs text-rose-400/90 mt-1">{errorMsg}</p>
            </div>
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="space-y-6">
            
            {/* Transcript Card */}
            <div className="p-5 rounded-2xl bg-slate-950/60 border border-slate-800">
              <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                <span className="font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                  <Mic className="w-3.5 h-3.5 text-indigo-400" />
                  Recognized Transcript
                </span>
                <span className="font-mono text-indigo-400">Hindi (hi-IN)</span>
              </div>
              <p className="text-lg font-medium text-slate-100 bg-slate-900/50 p-3.5 rounded-xl border border-slate-800/80">
                "{result.transcript}"
              </p>
            </div>

            {/* Answer & Grounding Card */}
            <div className="p-5 rounded-2xl bg-slate-950/60 border border-slate-800">
              <div className="flex items-center justify-between mb-3">
                <span className="font-semibold text-xs uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5 text-cyan-400" />
                  Grounded Answer
                </span>
                <div className="flex items-center gap-2">
                  {result.grounded ? (
                    <span className="flex items-center gap-1 text-xs font-mono bg-emerald-500/10 text-emerald-400 px-2.5 py-1 rounded-full border border-emerald-500/20">
                      <CheckCircle2 className="w-3.5 h-3.5" /> Grounded
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-xs font-mono bg-amber-500/10 text-amber-400 px-2.5 py-1 rounded-full border border-amber-500/20">
                      <AlertCircle className="w-3.5 h-3.5" /> Low Grounding / Refusal
                    </span>
                  )}
                  <span className="text-xs font-mono text-slate-400 bg-slate-900 px-2 py-1 rounded-md border border-slate-800">
                    Conf: {result.confidence}
                  </span>
                </div>
              </div>
              <p className="text-base text-slate-200 leading-relaxed bg-slate-900/80 p-4 rounded-xl border border-slate-800">
                {result.answer}
              </p>
            </div>

            {/* Latency Breakdown Grid */}
            <div className="p-5 rounded-2xl bg-slate-950/60 border border-slate-800">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3 flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-purple-400" />
                Real Measured Latency Breakdown
              </h4>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 font-mono text-center">
                <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-500 block">STT</span>
                  <span className="text-sm font-bold text-indigo-400">{result.latency.stt_ms} ms</span>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-500 block">Retrieval</span>
                  <span className="text-sm font-bold text-cyan-400">{result.latency.retrieval_ms} ms</span>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-500 block">Context</span>
                  <span className="text-sm font-bold text-slate-300">{result.latency.context_ms} ms</span>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-500 block">LLM / Fallback</span>
                  <span className="text-sm font-bold text-amber-400">{result.latency.llm_ms} ms</span>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-500 block">Grounding</span>
                  <span className="text-sm font-bold text-emerald-400">{result.latency.grounding_ms} ms</span>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-700/80">
                  <span className="text-[10px] text-slate-400 block">Total Server</span>
                  <span className="text-sm font-bold text-purple-400">{result.latency.total_ms} ms</span>
                </div>
              </div>
            </div>

            {/* Retrieved Sources Section */}
            {result.sources && result.sources.length > 0 && (
              <div className="p-5 rounded-2xl bg-slate-950/60 border border-slate-800">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3 flex items-center gap-1.5">
                  <Database className="w-3.5 h-3.5 text-cyan-400" />
                  Top Retrieved Context Chunks ({result.sources.length})
                </h4>
                <div className="space-y-2.5">
                  {result.sources.map((source, idx) => (
                    <div key={idx} className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-xs">
                      <div className="flex items-center justify-between text-slate-400 mb-1 font-mono">
                        <span>Chunk: {source.chunk_id}</span>
                        <span className="text-indigo-400">Score: {source.score}</span>
                      </div>
                      <p className="text-slate-300 line-clamp-2">{source.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        )}

        {/* Footer */}
        <div className="mt-8 text-center text-xs text-slate-500 flex items-center justify-center gap-2 border-t border-slate-800 pt-4">
          <Cpu className="w-3.5 h-3.5" />
          <span>Voice RAG Goa Hackathon 2026 • Stage 5B Implementation Complete</span>
        </div>

      </div>
    </div>
  )
}
