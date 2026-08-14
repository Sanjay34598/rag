import { useState, useEffect, useRef } from 'react'
import { 
  Mic, Square, Loader2, CheckCircle2, AlertCircle, 
  Search, FileText, Clock, Database, RefreshCw, ShieldCheck
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
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop())
        const finalMimeType = mediaRecorder.mimeType || mimeType || 'audio/webm'
        const audioBlob = new Blob(audioChunksRef.current, { type: finalMimeType })
        await submitVoiceQuery(audioBlob, finalMimeType)
      }

      // Record in 250ms chunks to ensure all audio data is captured
      mediaRecorder.start(250)
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
    if (!audioBlob || audioBlob.size === 0) {
      setErrorMsg('No audio recorded. Please try speaking again.')
      setUiState('ERROR')
      return
    }

    const cleanMime = mimeType.split(';')[0].trim()
    let extension = 'webm'
    if (cleanMime.includes('wav')) extension = 'wav'
    else if (cleanMime.includes('mp3') || cleanMime.includes('mpeg')) extension = 'mp3'
    else if (cleanMime.includes('ogg')) extension = 'ogg'
    else if (cleanMime.includes('mp4') || cleanMime.includes('m4a')) extension = 'm4a'

    const formData = new FormData()
    const filename = `recording.${extension}`
    formData.append('file', audioBlob, filename)

    try {
      let res: Response
      try {
        res = await fetch('/api/v1/voice/query', {
          method: 'POST',
          body: formData,
        })
      } catch (networkErr: any) {
        res = await fetch('http://127.0.0.1:8000/api/v1/voice/query', {
          method: 'POST',
          body: formData,
        })
      }

      const data = await res.json().catch(() => ({}))

      if (!res.ok) {
        let detailMsg = ''
        if (typeof data.detail === 'string') {
          detailMsg = data.detail
        } else if (Array.isArray(data.detail)) {
          detailMsg = data.detail.map((d: any) => d.msg || JSON.stringify(d)).join(', ')
        } else if (data.error) {
          detailMsg = data.error
        } else {
          detailMsg = res.statusText || 'Unknown Error'
        }

        if (res.status === 400) throw new Error(`Bad Request (400): ${detailMsg}`)
        if (res.status === 404) throw new Error(`Endpoint Not Found (404): ${detailMsg}`)
        if (res.status === 422) throw new Error(`Validation Error (422): ${detailMsg}`)
        if (res.status === 500) throw new Error(`Backend Error (500): ${detailMsg}`)
        if (res.status >= 502 && res.status <= 504) throw new Error(`Service Error (${res.status}): ${detailMsg}`)
        throw new Error(`Voice RAG API returned ${res.status}: ${detailMsg}`)
      }

      setResult(data)
      setUiState('SUCCESS')
    } catch (err: any) {
      console.error('Voice Query API Error:', err)
      const msg = err.name === 'TypeError' || err.message?.includes('fetch') 
        ? 'Failed to connect to Voice RAG API endpoint.' 
        : err.message
      setErrorMsg(msg)
      setUiState('ERROR')
    }
  }

  const runTextQuery = async (queryToRun: string) => {
    if (!queryToRun.trim()) return

    setUiState('PROCESSING')
    setErrorMsg(null)
    setResult(null)

    try {
      let res: Response
      try {
        res = await fetch('/api/v1/rag/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: queryToRun }),
        })
      } catch (networkErr: any) {
        res = await fetch('http://127.0.0.1:8000/api/v1/rag/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: queryToRun }),
        })
      }

      const data = await res.json().catch(() => ({}))

      if (!res.ok) {
        let detailMsg = ''
        if (typeof data.detail === 'string') {
          detailMsg = data.detail
        } else if (Array.isArray(data.detail)) {
          detailMsg = data.detail.map((d: any) => d.msg || JSON.stringify(d)).join(', ')
        } else if (data.error) {
          detailMsg = data.error
        } else {
          detailMsg = res.statusText || 'Unknown Error'
        }

        if (res.status === 400) throw new Error(`Bad Request (400): ${detailMsg}`)
        if (res.status === 404) throw new Error(`Endpoint Not Found (404): ${detailMsg}`)
        if (res.status === 422) throw new Error(`Validation Error (422): ${detailMsg}`)
        if (res.status === 500) throw new Error(`Backend Error (500): ${detailMsg}`)
        if (res.status >= 502 && res.status <= 504) throw new Error(`Service Error (${res.status}): ${detailMsg}`)
        throw new Error(`Text RAG API returned ${res.status}: ${detailMsg}`)
      }

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
      console.error('Text Query API Error:', err)
      const msg = err.name === 'TypeError' || err.message?.includes('fetch') 
        ? 'Failed to connect to Voice RAG API endpoint.' 
        : err.message
      setErrorMsg(msg)
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
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 flex flex-col items-center justify-between p-4 sm:p-6 md:p-8 font-sans">
      
      {/* Main Responsive Layout Wrapper */}
      <div className="max-w-4xl w-full space-y-6">
        
        {/* Header */}
        <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-200 dark:border-slate-800">
          <div>
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900 dark:text-white flex items-center gap-2">
              Voice RAG
            </h1>
            <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
              Ask questions in Hindi and get answers grounded in trusted knowledge.
            </p>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-auto">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 dark:bg-emerald-950/50 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400 text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              System Ready
            </span>
          </div>
        </header>

        {/* Main Search & Voice Control Hub */}
        <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 sm:p-6 shadow-sm space-y-4">
          
          {/* Integrated Search Input + Mic Button Bar */}
          <form onSubmit={handleTextSubmit} className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
            <div className="relative flex-1">
              <input
                type="text"
                value={textQuery}
                onChange={(e) => setTextQuery(e.target.value)}
                disabled={uiState === 'RECORDING' || uiState === 'PROCESSING'}
                placeholder="अपना प्रश्न यहाँ लिखें (Or speak in Hindi)..."
                className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-lg px-4 py-2.5 text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:opacity-50"
              />
            </div>

            <div className="flex items-center gap-2 shrink-0">
              {uiState === 'RECORDING' ? (
                <button
                  type="button"
                  onClick={stopRecording}
                  aria-label="Stop recording speech"
                  className="px-4 py-2.5 bg-rose-600 hover:bg-rose-700 text-white rounded-lg text-sm font-medium transition flex items-center gap-2 shadow-sm cursor-pointer"
                >
                  <Square className="w-4 h-4 fill-current" />
                  <span>Stop ({formatTime(recordingTime)})</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={startRecording}
                  disabled={uiState === 'PROCESSING'}
                  aria-label="Microphone input - speak in Hindi"
                  className="px-3.5 py-2.5 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-200 rounded-lg text-sm font-medium transition flex items-center gap-2 disabled:opacity-50 cursor-pointer"
                  title="Click to record voice query in Hindi"
                >
                  <Mic className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
                  <span className="hidden sm:inline">Voice</span>
                </button>
              )}

              <button
                type="submit"
                disabled={uiState === 'PROCESSING' || uiState === 'RECORDING' || !textQuery.trim()}
                className="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition flex items-center gap-2 shadow-sm shrink-0 cursor-pointer"
              >
                <Search className="w-4 h-4" />
                <span>Search</span>
              </button>
            </div>
          </form>

          {/* Active Voice Recording Status Bar */}
          {uiState === 'RECORDING' && (
            <div className="p-3 bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900 rounded-lg flex items-center justify-between text-xs text-rose-700 dark:text-rose-300">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
                <span className="font-semibold">● Recording active ({formatTime(recordingTime)})</span>
                <span className="hidden sm:inline text-rose-600 dark:text-rose-400">— Speak your question in Hindi</span>
              </div>
              <button
                onClick={stopRecording}
                className="text-xs font-semibold underline hover:text-rose-900 dark:hover:text-white"
              >
                Click to stop
              </button>
            </div>
          )}

          {/* Sample Questions Row */}
          {uiState !== 'RECORDING' && uiState !== 'PROCESSING' && (
            <div className="pt-3 border-t border-slate-100 dark:border-slate-800/80 text-xs flex flex-wrap items-center gap-2 text-slate-500 dark:text-slate-400">
              <span className="font-medium text-slate-700 dark:text-slate-300">Try a sample question:</span>
              {sampleQueries.map((sample, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSampleClick(sample)}
                  className="hover:text-indigo-600 dark:hover:text-indigo-400 underline decoration-slate-300 dark:decoration-slate-700 underline-offset-2 transition-colors cursor-pointer"
                >
                  "{sample}"
                </button>
              ))}
            </div>
          )}

        </section>

        {/* Processing State Notice */}
        {uiState === 'PROCESSING' && (
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
            <div className="flex items-center gap-3 text-sm font-medium text-slate-700 dark:text-slate-300">
              <Loader2 className="w-4 h-4 text-indigo-600 dark:text-indigo-400 animate-spin" />
              <span>Processing your question...</span>
            </div>
            <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500 dark:text-slate-400 font-mono">
              <span className="flex items-center gap-1"><CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" /> Speech recognized</span>
              <span>•</span>
              <span className="flex items-center gap-1"><CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" /> Searching knowledge</span>
              <span>•</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" /> Generating answer</span>
            </div>
          </div>
        )}

        {/* Error Alert Card */}
        {errorMsg && (
          <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 text-rose-800 dark:text-rose-200 flex items-start gap-3 text-sm">
            <AlertCircle className="w-5 h-5 text-rose-600 dark:text-rose-400 shrink-0 mt-0.5" />
            <div>
              <h4 className="font-semibold text-rose-900 dark:text-rose-100">Processing Notice</h4>
              <p className="text-xs text-rose-700 dark:text-rose-300 mt-1">{errorMsg}</p>
            </div>
          </div>
        )}

        {/* Results Container */}
        {result && (
          <div className="space-y-5">
            
            {/* Reset / Ask Another Question Button */}
            <div className="flex justify-end">
              <button
                onClick={resetDemo}
                className="text-xs bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300 px-3 py-1.5 rounded-lg transition shadow-sm flex items-center gap-1.5 font-medium cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Ask another question</span>
              </button>
            </div>

            {/* Answer Card */}
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
                <h3 className="text-sm font-semibold text-slate-900 dark:text-white uppercase tracking-wider flex items-center gap-2">
                  <FileText className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
                  Answer
                </h3>
                <div className="flex items-center gap-2 text-xs">
                  {result.grounded ? (
                    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-50 dark:bg-emerald-950/60 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400 font-medium">
                      <CheckCircle2 className="w-3.5 h-3.5" /> Grounded
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-amber-50 dark:bg-amber-950/60 border border-amber-200 dark:border-amber-800 text-amber-700 dark:text-amber-400 font-medium">
                      <ShieldCheck className="w-3.5 h-3.5" /> Grounding Protection Active
                    </span>
                  )}
                  <span className="font-mono text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded">
                    Confidence: {(result.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {result.grounded ? (
                <p className="text-base sm:text-lg text-slate-800 dark:text-slate-100 leading-relaxed font-normal">
                  {result.answer}
                </p>
              ) : (
                <div className="p-4 rounded-lg bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/60 text-slate-700 dark:text-slate-300 space-y-1">
                  <p className="text-sm font-medium text-amber-900 dark:text-amber-200">
                    "I couldn't verify this answer from the available knowledge."
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    The system avoids generating unsupported answers when retrieved evidence is insufficient.
                  </p>
                </div>
              )}
            </div>

            {/* Recognized Speech Card */}
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-sm space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                <span className="font-medium text-slate-700 dark:text-slate-300">Recognized speech</span>
                <span className="font-mono">Language: Hindi (hi-IN)</span>
              </div>
              <p className="text-sm sm:text-base font-medium text-slate-800 dark:text-slate-200">
                "{result.transcript}"
              </p>
            </div>

            {/* Performance Metrics */}
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                <span className="font-semibold text-slate-900 dark:text-white uppercase tracking-wider flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
                  Performance
                </span>
                <span className="text-[11px]">Target: &lt;200 ms for local retrieval stage</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
                <div className="p-3 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg flex items-center justify-between">
                  <span className="text-slate-500">Retrieval</span>
                  <span className="font-bold text-slate-900 dark:text-slate-100">{result.latency.retrieval_ms} ms</span>
                </div>
                <div className="p-3 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg flex items-center justify-between">
                  <span className="text-slate-500">Gemini</span>
                  <span className="font-bold text-slate-900 dark:text-slate-100">{result.latency.llm_ms} ms</span>
                </div>
                <div className="p-3 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg flex items-center justify-between">
                  <span className="text-slate-500">Total</span>
                  <span className="font-bold text-indigo-600 dark:text-indigo-400">{result.latency.total_ms} ms</span>
                </div>
              </div>
            </div>

            {/* Sources List */}
            {result.sources && result.sources.length > 0 && (
              <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
                <h4 className="text-xs font-semibold text-slate-900 dark:text-white uppercase tracking-wider flex items-center gap-1.5">
                  <Database className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
                  Sources ({result.sources.length})
                </h4>
                <div className="space-y-2.5">
                  {result.sources.map((source, idx) => (
                    <div key={idx} className="p-3.5 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg text-xs space-y-1">
                      <div className="flex items-center justify-between font-mono text-slate-400">
                        <span className="font-semibold text-slate-700 dark:text-slate-300">Source {idx + 1} • {source.chunk_id}</span>
                        <span>Score: {source.score}</span>
                      </div>
                      <p className="text-slate-700 dark:text-slate-300 leading-relaxed">{source.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        )}

        {/* Footer */}
        <footer className="pt-6 border-t border-slate-200 dark:border-slate-800 text-center text-xs text-slate-400 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Voice RAG • Hackathon Production System</span>
          <span className="font-mono text-[11px]">Sarvam STT + FAISS/BM25 Hybrid + Gemini 2.5 Flash</span>
        </footer>

      </div>
    </div>
  )
}
