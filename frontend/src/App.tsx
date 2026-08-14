import { useState, useEffect, useRef } from 'react'
import { 
  Mic, Square, Loader2, CheckCircle2, AlertCircle, 
  Search, FileText, Cpu, Clock, ShieldCheck, Database, Volume2,
  RefreshCw, Sparkles, HelpCircle, Layers, Zap
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
  language_code?: string
  language_probability?: number
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

  const getSupportedMimeType = () => {
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
      'audio/wav'
    ]
    for (const type of candidates) {
      if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(type)) {
        return type
      }
    }
    return ''
  }

  const startRecording = async () => {
    setErrorMsg(null)
    setResult(null)
    audioChunksRef.current = []

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mimeType = getSupportedMimeType()
      const options = mimeType ? { mimeType } : {}
      const mediaRecorder = new MediaRecorder(stream, options)
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        // Stop stream tracks
        stream.getTracks().forEach((track) => track.stop())

        const finalMimeType = mediaRecorder.mimeType || mimeType || 'audio/webm'
        const audioBlob = new Blob(audioChunksRef.current, { type: finalMimeType })
        await submitVoiceQuery(audioBlob, finalMimeType)
      }

      mediaRecorder.start()
      setUiState('RECORDING')
      setRecordingTime(0)

      timerRef.current = window.setInterval(() => {
        setRecordingTime((prev) => {
          if (prev >= 29) {
            stopRecording()
            return 30
          }
          return prev + 1
        })
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

  const runTextQuery = async (queryToRun: string) => {
    if (!queryToRun.trim()) return

    setUiState('PROCESSING')
    setErrorMsg(null)
    setResult(null)

    try {
      const endpoints = ['/api/v1/rag/query', 'http://127.0.0.1:8000/api/v1/rag/query']
      let response: Response | null = null

      for (const url of endpoints) {
        try {
          const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: queryToRun }),
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
        transcript: queryToRun,
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

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    runTextQuery(textQuery)
  }

  const handleSampleClick = (sampleQuery: string) => {
    setTextQuery(sampleQuery)
    runTextQuery(sampleQuery)
  }

  const resetDemo = () => {
    if (timerRef.current) clearInterval(timerRef.current)
    setUiState('IDLE')
    setRecordingTime(0)
    setResult(null)
    setErrorMsg(null)
    setTextQuery('')
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  const sampleQueries = [
    "कॉर्पोरेशन क्या है?",
    "ईमानदारी की परिभाषा",
    "वायुमंडलीय दबाव की परिभाषा"
  ]

  return (
    <div className="min-h-screen bg-[#070c14] text-slate-100 flex flex-col items-center justify-start p-4 sm:p-6 font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      
      {/* Container */}
      <div className="max-w-4xl w-full bg-slate-900/80 border border-slate-800 rounded-3xl p-5 sm:p-8 shadow-2xl backdrop-blur-xl">
        
        {/* Header / Hero */}
        <header className="pb-6 mb-8 border-b border-slate-800/80">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-center gap-3.5">
              <div className="w-11 h-11 rounded-2xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0 shadow-inner">
                <Volume2 className="w-5 h-5" />
              </div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
                  Voice RAG
                </h1>
                <p className="text-xs text-slate-400 mt-0.5">
                  Ask questions in Hindi. Get grounded answers from trusted knowledge.
                </p>
              </div>
            </div>

            {/* System Status */}
            <div className="flex items-center gap-2 self-start md:self-auto">
              <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                System Ready
              </span>
            </div>
          </div>

          {/* Tech Stack Badges */}
          <div className="flex flex-wrap items-center gap-2 mt-4 pt-4 border-t border-slate-800/50 text-[11px] font-mono text-slate-400">
            <span className="px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-slate-300 flex items-center gap-1.5">
              <Zap className="w-3 h-3 text-indigo-400" />
              Sarvam • saaras:v3
            </span>
            <span className="px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-slate-300 flex items-center gap-1.5">
              <Database className="w-3 h-3 text-cyan-400" />
              FAISS + BM25 Hybrid
            </span>
            <span className="px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-slate-300 flex items-center gap-1.5">
              <Cpu className="w-3 h-3 text-amber-400" />
              Gemini 2.5 Flash
            </span>
            <span className="px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-slate-300 flex items-center gap-1.5">
              <ShieldCheck className="w-3 h-3 text-emerald-400" />
              Grounding Active
            </span>
          </div>
        </header>

        {/* Main Voice Control Section */}
        <section className="flex flex-col items-center justify-center p-6 sm:p-8 bg-slate-950/80 border border-slate-800/90 rounded-2xl mb-8 relative overflow-hidden">
          
          {/* Microphone Main Action Button */}
          {uiState === 'RECORDING' ? (
            <div className="flex flex-col items-center gap-3">
              <button
                onClick={stopRecording}
                aria-label="Stop recording speech"
                className="w-24 h-24 rounded-full bg-rose-600 hover:bg-rose-500 text-white flex flex-col items-center justify-center gap-1 shadow-lg shadow-rose-600/30 transition-all transform hover:scale-105 active:scale-95 border-4 border-rose-400/40 cursor-pointer"
              >
                <Square className="w-8 h-8 fill-current" />
                <span className="text-[10px] font-bold uppercase tracking-wider">Stop</span>
              </button>

              {/* Animated Waveform Visualizer */}
              <div className="flex items-center gap-1.5 h-6 my-1">
                <span className="w-1.5 bg-rose-400 rounded-full animate-waveform-1" />
                <span className="w-1.5 bg-rose-400 rounded-full animate-waveform-2" />
                <span className="w-1.5 bg-rose-400 rounded-full animate-waveform-3" />
                <span className="w-1.5 bg-rose-400 rounded-full animate-waveform-4" />
                <span className="w-1.5 bg-rose-400 rounded-full animate-waveform-5" />
              </div>

              <div className="text-rose-400 font-mono text-sm font-semibold flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-rose-500 animate-ping" />
                <span>Listening... ({formatTime(recordingTime)})</span>
              </div>
              <span className="text-xs text-slate-400">Tap to stop</span>
            </div>
          ) : uiState === 'PROCESSING' ? (
            <div className="flex flex-col items-center gap-3 py-2">
              <div className="w-24 h-24 rounded-full bg-indigo-500/10 border border-indigo-500/40 text-indigo-400 flex items-center justify-center shadow-inner">
                <Loader2 className="w-10 h-10 animate-spin" />
              </div>
              <p className="text-sm text-indigo-300 font-medium">Processing your question...</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <button
                onClick={startRecording}
                aria-label="Microphone input - Tap to speak in Hindi"
                className="w-24 h-24 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white flex flex-col items-center justify-center gap-1 shadow-lg shadow-indigo-600/30 transition-all transform hover:scale-105 active:scale-95 border-4 border-indigo-400/30 cursor-pointer"
              >
                <Mic className="w-9 h-9" />
                <span className="text-[10px] font-bold uppercase tracking-wider">Speak</span>
              </button>
              <p className="text-xs font-medium text-slate-300">Tap to speak</p>
            </div>
          )}

          {/* Quick Hindi Demo Sample Queries */}
          {uiState !== 'RECORDING' && uiState !== 'PROCESSING' && (
            <div className="mt-8 w-full max-w-lg pt-6 border-t border-slate-800/80">
              <div className="flex items-center justify-between mb-2.5">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                  Try a sample question
                </span>
                <span className="text-[11px] text-slate-500">Instant Demo</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {sampleQueries.map((sample, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSampleClick(sample)}
                    className="text-xs bg-slate-900 hover:bg-slate-800 border border-slate-700/80 hover:border-indigo-500/50 text-slate-200 px-3 py-1.5 rounded-lg transition-colors text-left font-medium"
                  >
                    "{sample}"
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Text Query Input Fallback */}
          {uiState !== 'RECORDING' && uiState !== 'PROCESSING' && (
            <form onSubmit={handleTextSubmit} className="mt-4 w-full max-w-lg flex items-center gap-2">
              <input
                type="text"
                value={textQuery}
                onChange={(e) => setTextQuery(e.target.value)}
                placeholder="या अपना प्रश्न यहाँ लिखें..."
                className="flex-1 bg-slate-900 border border-slate-700/80 rounded-xl px-4 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              />
              <button
                type="submit"
                disabled={!textQuery.trim()}
                className="bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 px-4 py-2.5 rounded-xl text-xs font-semibold transition border border-slate-700 flex items-center gap-1.5 shrink-0"
              >
                <Search className="w-3.5 h-3.5" />
                <span>Search</span>
              </button>
            </form>
          )}

        </section>

        {/* Processing 4-Stage Stepper View */}
        {uiState === 'PROCESSING' && (
          <div className="p-6 rounded-2xl bg-slate-950/90 border border-slate-800 mb-8 space-y-4">
            <div className="flex items-center justify-between text-xs text-slate-400 pb-3 border-b border-slate-800/80">
              <span className="font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-400" />
                Processing through 4-stage RAG pipeline...
              </span>
              <span className="font-mono text-indigo-400 text-[11px] animate-pulse">Active</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-slate-900/90 border border-indigo-500/30 flex flex-col gap-1">
                <span className="font-semibold text-indigo-300 flex items-center gap-1">
                  ① Sarvam STT
                </span>
                <span className="text-[11px] text-slate-400">Transcribing speech</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/90 border border-cyan-500/30 flex flex-col gap-1">
                <span className="font-semibold text-cyan-300 flex items-center gap-1">
                  ② FAISS + BM25
                </span>
                <span className="text-[11px] text-slate-400">Searching knowledge</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/90 border border-amber-500/30 flex flex-col gap-1">
                <span className="font-semibold text-amber-300 flex items-center gap-1">
                  ③ Gemini 2.5 Flash
                </span>
                <span className="text-[11px] text-slate-400">Generating answer</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/90 border border-emerald-500/30 flex flex-col gap-1">
                <span className="font-semibold text-emerald-300 flex items-center gap-1">
                  ④ Grounding & Safety
                </span>
                <span className="text-[11px] text-slate-400">Validating response</span>
              </div>
            </div>
          </div>
        )}

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

            {/* Reset Button */}
            <div className="flex justify-end">
              <button
                onClick={resetDemo}
                className="text-xs bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 px-3.5 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 font-medium"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Ask another question</span>
              </button>
            </div>

            {/* Grounded Answer Hero Card (or Refusal Guardrail Card) */}
            <div className="p-6 rounded-2xl bg-slate-950/90 border border-slate-800/90">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-4 pb-3 border-b border-slate-800/80">
                <span className="font-semibold text-xs uppercase tracking-wider text-slate-400 flex items-center gap-2">
                  <FileText className="w-4 h-4 text-cyan-400" />
                  {result.grounded ? "Grounded Answer" : "Answer Not Verified"}
                </span>

                <div className="flex items-center gap-2">
                  {result.grounded ? (
                    <span className="flex items-center gap-1.5 text-xs font-medium bg-emerald-500/10 text-emerald-400 px-3 py-1 rounded-full border border-emerald-500/30">
                      <CheckCircle2 className="w-3.5 h-3.5" /> GROUNDED
                    </span>
                  ) : (
                    <span className="flex items-center gap-1.5 text-xs font-medium bg-amber-500/10 text-amber-400 px-3 py-1 rounded-full border border-amber-500/30">
                      <AlertCircle className="w-3.5 h-3.5" /> Grounding Protection Active
                    </span>
                  )}
                  <span className="text-xs font-mono text-slate-400 bg-slate-900 px-2.5 py-1 rounded-md border border-slate-800">
                    Confidence: {(result.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Answer Content */}
              {result.grounded ? (
                <p className="text-lg text-slate-100 leading-relaxed bg-slate-900/60 p-5 rounded-xl border border-slate-800/80 font-normal">
                  {result.answer}
                </p>
              ) : (
                <div className="p-5 rounded-xl bg-amber-500/5 border border-amber-500/20 text-slate-300 space-y-2">
                  <p className="text-base text-amber-200 font-medium">
                    "I couldn't verify this answer from the available knowledge."
                  </p>
                  <p className="text-xs text-slate-400">
                    The system avoids generating an unsupported answer when the retrieved evidence is insufficient.
                  </p>
                </div>
              )}
            </div>

            {/* Recognized Speech / Transcript Card */}
            <div className="p-5 rounded-2xl bg-slate-950/70 border border-slate-800/80">
              <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                <span className="font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Mic className="w-3.5 h-3.5 text-indigo-400" />
                  Recognized Speech
                </span>
                <div className="flex items-center gap-2 font-mono text-indigo-400 text-xs">
                  <span>Language: {result.language_code || 'hi-IN'}</span>
                  {result.language_probability !== undefined && result.language_probability !== null && (
                    <span className="text-slate-400">| Confidence: {(result.language_probability * 100).toFixed(0)}%</span>
                  )}
                </div>
              </div>
              <p className="text-base font-medium text-slate-200 bg-slate-900/50 p-3.5 rounded-xl border border-slate-800/80">
                "{result.transcript}"
              </p>
            </div>

            {/* Pipeline Performance / Latency Breakdown */}
            <div className="p-5 rounded-2xl bg-slate-950/70 border border-slate-800/80">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-purple-400" />
                  Pipeline Performance
                </h4>
                <span className="text-[11px] text-slate-500">
                  Target: &lt;200 ms for local hybrid retrieval stage
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                
                {/* Local Retrieval */}
                <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col justify-between gap-2">
                  <span className="text-[11px] text-slate-400 uppercase tracking-wider font-semibold">Local Retrieval</span>
                  <div className="flex items-baseline justify-between font-mono">
                    <span className="text-slate-400">Hybrid Search:</span>
                    <span className="text-base font-bold text-cyan-400">{result.latency.retrieval_ms} ms</span>
                  </div>
                </div>

                {/* Cloud Services */}
                <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col justify-between gap-2">
                  <span className="text-[11px] text-slate-400 uppercase tracking-wider font-semibold">Cloud Services</span>
                  <div className="space-y-1 font-mono">
                    <div className="flex items-baseline justify-between">
                      <span className="text-slate-400">Sarvam STT:</span>
                      <span className="text-sm font-semibold text-indigo-400">{result.latency.stt_ms} ms</span>
                    </div>
                    <div className="flex items-baseline justify-between">
                      <span className="text-slate-400">Gemini 2.5 Flash:</span>
                      <span className="text-sm font-semibold text-amber-400">{result.latency.llm_ms} ms</span>
                    </div>
                  </div>
                </div>

                {/* Total Server Latency */}
                <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-700/80 flex flex-col justify-between gap-2">
                  <span className="text-[11px] text-slate-400 uppercase tracking-wider font-semibold">Total Server Latency</span>
                  <div className="flex items-baseline justify-between font-mono">
                    <span className="text-slate-400">Total:</span>
                    <span className="text-base font-bold text-purple-400">{result.latency.total_ms} ms</span>
                  </div>
                </div>

              </div>
            </div>

            {/* Retrieved Sources Section */}
            {result.sources && result.sources.length > 0 && (
              <div className="p-5 rounded-2xl bg-slate-950/70 border border-slate-800/80">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
                  <Database className="w-3.5 h-3.5 text-cyan-400" />
                  Retrieved Sources ({result.sources.length})
                </h4>
                <div className="space-y-2.5">
                  {result.sources.map((source, idx) => (
                    <div key={idx} className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800/80 text-xs space-y-1">
                      <div className="flex items-center justify-between text-slate-400 font-mono">
                        <span className="font-semibold text-slate-300">SOURCE {(idx + 1).toString().padStart(2, '0')} • {source.chunk_id}</span>
                        <span className="text-indigo-400">Score: {source.score}</span>
                      </div>
                      <p className="text-slate-300 leading-relaxed">{source.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        )}

        {/* Footer */}
        <footer className="mt-8 text-center text-xs text-slate-500 flex items-center justify-center gap-2 border-t border-slate-800/80 pt-4">
          <Cpu className="w-3.5 h-3.5" />
          <span>Voice RAG • Hackathon Production System</span>
        </footer>

      </div>
    </div>
  )
}
