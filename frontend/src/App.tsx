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
  language?: string
  query_id?: number
  score: number
  text: string
  source_lang?: string
  target_lang?: string
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
  const [selectedLang, setSelectedLang] = useState<string>('unknown')
  const [result, setResult] = useState<VoiceQueryResponse | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)

  const languageOptions = [
    { code: 'unknown', label: 'Auto Detect' },
    { code: 'hi-IN', label: 'Hindi (hi-IN)' },
    { code: 'en-IN', label: 'English (en-IN)' },
    { code: 'te-IN', label: 'Telugu (te-IN)' }
  ]

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
      setResult(null)
      setErrorMsg(null)
    }
  }

  const submitVoiceQuery = async (audioBlob: Blob, mimeType: string) => {
    setResult(null)
    setErrorMsg(null)
    setUiState('PROCESSING')

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
    if (selectedLang && selectedLang !== 'unknown') {
      formData.append('language_code', selectedLang)
      formData.append('language', selectedLang)
    }

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
      setResult(null)
      const msg = err.name === 'TypeError' || err.message?.includes('fetch') 
        ? 'Failed to connect to Voice RAG API endpoint.' 
        : err.message
      setErrorMsg(msg)
      setUiState('ERROR')
    }
  }

  const runTextQuery = async (queryToRun: string, targetLangOverride?: string) => {
    if (!queryToRun.trim()) return

    setUiState('PROCESSING')
    setErrorMsg(null)
    setResult(null)

    const langToUse = targetLangOverride || selectedLang
    const reqBody: any = { query: queryToRun }
    if (langToUse && langToUse !== 'unknown') {
      reqBody.language_code = langToUse
      reqBody.language = langToUse
    }

    try {
      let res: Response
      try {
        res = await fetch('/api/v1/rag/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(reqBody),
        })
      } catch (networkErr: any) {
        res = await fetch('http://127.0.0.1:8000/api/v1/rag/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(reqBody),
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
        language_code: data.language_code || (langToUse !== 'unknown' ? langToUse : 'hi-IN'),
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
      setResult(null)
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

  const handleSampleClick = (sampleQuery: string, sampleLang?: string) => {
    setTextQuery(sampleQuery)
    if (sampleLang) {
      setSelectedLang(sampleLang)
    }
    runTextQuery(sampleQuery, sampleLang)
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

  const getLanguageDisplayLabel = (code?: string) => {
    if (code === 'hi-IN') return 'Hindi (hi-IN)'
    if (code === 'en-IN') return 'English (en-IN)'
    if (code === 'te-IN') return 'Telugu (te-IN)'
    return code || 'Auto Detected'
  }

  const sampleQueries = [
    { text: "कॉर्पोरेशन क्या है?", lang: "hi-IN", label: "Hindi: Corporation" },
    { text: "What is a corporation?", lang: "en-IN", label: "English: Corporation" },
    { text: "కార్పొరేషన్ అంటే ఏమిటి?", lang: "te-IN", label: "Telugu: Corporation" }
  ]

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col justify-between font-sans selection:bg-emerald-500/20">
      
      {/* Container: max-width 1400px wide layout */}
      <div className="max-w-[1400px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6 flex-1">
        
        {/* HEADER */}
        <header className="flex items-center justify-between pb-5 border-b border-zinc-800">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-zinc-100 flex items-center gap-2">
              Voice RAG
            </h1>
            <p className="text-xs text-zinc-400 mt-0.5">
              Multilingual knowledge assistant (Hindi • English • Telugu)
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/60 border border-emerald-800 text-emerald-400 text-xs font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              System Ready
            </span>
          </div>
        </header>

        {/* MAIN RESPONSIVE TWO-COLUMN GRID */}
        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,2.1fr)_minmax(320px,0.9fr)] gap-6 lg:gap-8 items-start">
          
          {/* ================================================== */}
          {/* LEFT COLUMN (65-70% Width) */}
          {/* ================================================== */}
          <div className="space-y-6">
            
            {/* 1. ASK A QUESTION */}
            <section className="bg-zinc-900 border border-zinc-800 rounded-lg p-5 space-y-4">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <h2 className="text-lg font-semibold tracking-tight text-zinc-100">
                    Ask a question
                  </h2>
                  <p className="text-xs text-zinc-400 mt-0.5">
                    Search trusted knowledge using text or your voice.
                  </p>
                </div>

                {/* Professional Language Selector */}
                <div className="flex items-center gap-1.5 bg-zinc-950 border border-zinc-800 rounded-md px-2.5 py-1 text-xs">
                  <span className="text-zinc-500 font-medium hidden sm:inline">Language:</span>
                  <select
                    value={selectedLang}
                    onChange={(e) => setSelectedLang(e.target.value)}
                    aria-label="Select target language"
                    className="bg-transparent text-zinc-200 focus:outline-none cursor-pointer font-medium text-xs"
                  >
                    {languageOptions.map((opt) => (
                      <option key={opt.code} value={opt.code} className="bg-zinc-900 text-zinc-200">
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Integrated Search Input + Buttons */}
              <form onSubmit={handleTextSubmit} className="relative flex items-center w-full bg-zinc-950 border border-zinc-800 rounded-lg focus-within:border-zinc-500 transition-colors">
                <input
                  type="text"
                  value={textQuery}
                  onChange={(e) => setTextQuery(e.target.value)}
                  disabled={uiState === 'RECORDING' || uiState === 'PROCESSING'}
                  placeholder="Type your question in Hindi, English, or Telugu..."
                  className="w-full bg-transparent px-4 py-3 text-sm sm:text-base text-zinc-100 placeholder-zinc-500 focus:outline-none disabled:opacity-50 hindi-text flex-1"
                />

                <div className="flex items-center gap-1.5 pr-2 shrink-0">
                  {uiState === 'RECORDING' ? (
                    <button
                      type="button"
                      onClick={stopRecording}
                      aria-label="Stop recording speech"
                      className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded text-xs font-medium transition flex items-center gap-1.5 cursor-pointer"
                    >
                      <Square className="w-3.5 h-3.5 fill-current" />
                      <span>Stop ({formatTime(recordingTime)})</span>
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={startRecording}
                      disabled={uiState === 'PROCESSING'}
                      aria-label="Microphone input - speak in your language"
                      className="p-2 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 rounded transition disabled:opacity-50 cursor-pointer"
                      title="Click to speak"
                    >
                      <Mic className="w-4 h-4" />
                    </button>
                  )}

                  <button
                    type="submit"
                    disabled={uiState === 'PROCESSING' || uiState === 'RECORDING' || !textQuery.trim()}
                    className="px-4 py-2 bg-zinc-100 hover:bg-white text-zinc-900 font-semibold text-xs rounded transition disabled:opacity-40 shrink-0 cursor-pointer"
                  >
                    Search
                  </button>
                </div>
              </form>

              {/* Active Recording Status Bar */}
              {uiState === 'RECORDING' && (
                <div className="p-3 bg-red-950/30 border border-red-900/60 rounded flex items-center justify-between text-xs text-red-400">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    <span className="font-medium">Listening active ({formatTime(recordingTime)})</span>
                    <span className="hidden sm:inline text-red-300">— Speak your question in {selectedLang === 'unknown' ? 'your language' : getLanguageDisplayLabel(selectedLang)}</span>
                  </div>
                  <button
                    onClick={stopRecording}
                    className="text-xs font-semibold underline hover:text-white cursor-pointer"
                  >
                    Stop
                  </button>
                </div>
              )}

              {/* Processing Loader */}
              {uiState === 'PROCESSING' && (
                <div className="p-3 bg-zinc-950 border border-zinc-800 rounded flex items-center gap-2 text-xs text-zinc-400">
                  <Loader2 className="w-4 h-4 animate-spin text-zinc-100" />
                  <span>Searching trusted knowledge...</span>
                </div>
              )}

              {/* Sample Questions */}
              {uiState !== 'RECORDING' && uiState !== 'PROCESSING' && (
                <div className="pt-2 border-t border-zinc-800/60 text-xs flex flex-wrap items-center gap-2 text-zinc-500">
                  <span className="font-medium text-zinc-400">Try a sample question:</span>
                  {sampleQueries.map((sample, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleSampleClick(sample.text, sample.lang)}
                      className="px-2.5 py-1 bg-zinc-950 hover:bg-zinc-800 border border-zinc-800 rounded text-zinc-300 hover:text-white transition cursor-pointer hindi-text"
                    >
                      {sample.label}: "{sample.text}"
                    </button>
                  ))}
                </div>
              )}
            </section>

            {/* Processing Error Alert */}
            {errorMsg && (
              <div className="p-4 rounded-lg bg-red-950/30 border border-red-900/60 text-red-400 flex items-start gap-3 text-sm">
                <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-zinc-100">Processing Notice</h4>
                  <p className="text-xs text-red-300 mt-1">{errorMsg}</p>
                </div>
              </div>
            )}

            {/* 2. ANSWER SECTION */}
            {result && (
              <section className="bg-zinc-900 border border-zinc-800 rounded-lg p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                  <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                    Answer
                  </h3>
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-mono text-zinc-400 bg-zinc-950 px-2 py-0.5 rounded border border-zinc-800">
                      Response Language: {getLanguageDisplayLabel(result.language_code)}
                    </span>
                    {result.grounded ? (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-950/60 border border-emerald-800 text-emerald-400 text-xs font-medium">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Grounded
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-amber-950/60 border border-amber-800 text-amber-400 text-xs font-medium">
                        <ShieldCheck className="w-3.5 h-3.5" /> Not verified
                      </span>
                    )}
                  </div>
                </div>

                {result.grounded ? (
                  <div className="space-y-2">
                    <p className="text-xl sm:text-2xl text-zinc-100 leading-[1.8] font-normal hindi-text">
                      {result.answer}
                    </p>
                    <p className="text-xs text-zinc-500 font-mono">
                      Confidence: {(result.confidence * 100).toFixed(0)}%
                    </p>
                  </div>
                ) : (
                  <div className="p-4 rounded bg-amber-950/20 border border-amber-900/40 text-amber-200 space-y-1">
                    <p className="text-base font-medium hindi-text">
                      "{result.answer}"
                    </p>
                  </div>
                )}

                {/* RECOGNIZED SPEECH */}
                {result.transcript && (
                  <div className="pt-4 border-t border-zinc-800 space-y-1">
                    <div className="flex items-center justify-between text-xs text-zinc-500">
                      <span className="font-medium text-zinc-400">Recognized speech</span>
                      <span className="font-mono">Speech Language: {getLanguageDisplayLabel(result.language_code)}</span>
                    </div>
                    <p className="text-sm sm:text-base font-medium text-zinc-200 hindi-text pt-0.5">
                      "{result.transcript}"
                    </p>
                  </div>
                )}
              </section>
            )}

            {/* 3. SOURCES */}
            {result && (
              <section className="bg-zinc-900 border border-zinc-800 rounded-lg p-5 space-y-3">
                <div className="flex items-center justify-between border-b border-zinc-800 pb-2.5">
                  <div>
                    <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                      Canonical Sources ({result.sources?.length || 0})
                    </h4>
                    <p className="text-[11px] text-zinc-500">
                      Original retrieved MSMARCO-XI dataset evidence
                    </p>
                  </div>
                  {result.sources && result.sources.length > 0 && (
                    <span className="text-[11px] font-mono text-zinc-400 bg-zinc-950 px-2 py-0.5 rounded border border-zinc-800">
                      Evidence Language: English
                    </span>
                  )}
                </div>

                {result.sources && result.sources.length > 0 ? (
                  <div className="space-y-2.5">
                    {result.sources.map((source, idx) => (
                      <div key={idx} className="p-3.5 bg-zinc-950 border border-zinc-800/80 rounded space-y-1.5 text-xs">
                        <div className="flex items-center justify-between font-mono text-zinc-500">
                          <span className="font-semibold text-zinc-300">
                            {String(idx + 1).padStart(2, '0')} &nbsp; Source • {source.chunk_id}
                          </span>
                          <div className="flex items-center gap-2 text-[11px]">
                            <span className="text-zinc-400 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
                              Evidence: {source.language === 'en' ? 'English' : source.language || 'English'}
                            </span>
                            <span>Relevance: {source.score}</span>
                          </div>
                        </div>
                        <p className="text-xs sm:text-sm text-zinc-400 leading-relaxed hindi-text pt-1">{source.text}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-3.5 bg-zinc-950 border border-zinc-800/80 rounded text-xs text-zinc-400">
                    No verified sources available.
                  </div>
                )}
              </section>
            )}

          </div>

          {/* ================================================== */}
          {/* RIGHT COLUMN (30-35% Width - Sidebar) */}
          {/* ================================================== */}
          <div className="space-y-6">

            {/* 1. VOICE SEARCH SIDEBAR PANEL */}
            <section className="bg-zinc-900 border border-zinc-800 rounded-lg p-5 space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-zinc-100">
                  Voice Search
                </h3>
                <p className="text-xs text-zinc-400 mt-0.5">
                  Click the microphone and speak your question in Hindi, English, or Telugu.
                </p>
              </div>

              {uiState === 'RECORDING' ? (
                <button
                  onClick={stopRecording}
                  className="w-full bg-red-950/40 hover:bg-red-900/50 border border-red-900/60 text-red-400 py-2.5 px-4 rounded text-xs font-medium transition cursor-pointer flex items-center justify-center gap-2"
                >
                  <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                  <span>● Recording ({formatTime(recordingTime)}) — Click to stop</span>
                </button>
              ) : uiState === 'PROCESSING' ? (
                <div className="w-full bg-zinc-950 border border-zinc-800 text-zinc-400 py-2.5 px-4 rounded text-xs font-medium flex items-center justify-center gap-2">
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-zinc-100" />
                  <span>Searching trusted knowledge...</span>
                </div>
              ) : (
                <button
                  onClick={startRecording}
                  className="w-full bg-zinc-950 hover:bg-zinc-800 border border-zinc-800 text-zinc-200 py-2.5 px-4 rounded text-xs font-medium transition cursor-pointer flex items-center justify-center gap-2"
                >
                  <Mic className="w-4 h-4 text-emerald-400" />
                  <span>🎙 Speak your question</span>
                </button>
              )}
            </section>

            {/* 2. PERFORMANCE PANEL */}
            {result && (
              <section className="bg-zinc-900 border border-zinc-800 rounded-lg p-5 space-y-3">
                <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
                  <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                    Performance
                  </h3>
                  <span className="text-[10px] text-zinc-500 font-mono">&lt;200 ms target</span>
                </div>
                <div className="space-y-2 text-xs font-mono">
                  <div className="flex items-center justify-between text-zinc-400">
                    <span>Retrieval</span>
                    <span className="font-semibold text-zinc-200">{result.latency.retrieval_ms} ms</span>
                  </div>
                  <div className="flex items-center justify-between text-zinc-400">
                    <span>Groq</span>
                    <span className="font-semibold text-zinc-200">{result.latency.llm_ms} ms</span>
                  </div>
                  <div className="flex items-center justify-between text-zinc-400 pt-1.5 border-t border-zinc-800/60">
                    <span>Total</span>
                    <span className="font-semibold text-emerald-400">{result.latency.total_ms} ms</span>
                  </div>
                </div>
              </section>
            )}

            {/* 3. TECHNICAL DETAILS PANEL */}
            <section className="bg-zinc-900 border border-zinc-800 rounded-lg p-5 space-y-3">
              <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                Technical Details
              </h3>
              <div className="space-y-2 text-xs text-zinc-400">
                <div className="flex items-center justify-between border-b border-zinc-800/50 pb-1.5">
                  <span className="text-zinc-500">Dense retrieval</span>
                  <span className="font-mono text-zinc-200">FAISS Vector Index</span>
                </div>
                <div className="flex items-center justify-between border-b border-zinc-800/50 pb-1.5">
                  <span className="text-zinc-500">Sparse retrieval</span>
                  <span className="font-mono text-zinc-200">BM25 (Rank-BM25)</span>
                </div>
                <div className="flex items-center justify-between border-b border-zinc-800/50 pb-1.5">
                  <span className="text-zinc-500">LLM</span>
                  <span className="font-mono text-zinc-200">Groq llama-3.1-8b-instant</span>
                </div>
                <div className="flex items-center justify-between border-b border-zinc-800/50 pb-1.5">
                  <span className="text-zinc-500">STT Service</span>
                  <span className="font-mono text-zinc-200">Sarvam saaras:v3</span>
                </div>
                <div className="flex items-center justify-between border-b border-zinc-800/50 pb-1.5">
                  <span className="text-zinc-500">Supported Languages</span>
                  <span className="font-mono text-zinc-200">hi-IN, en-IN, te-IN</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-zinc-500">Retrieved sources</span>
                  <span className="font-mono text-zinc-200">
                    {result ? `${result.sources?.length || 0} Chunks` : "0 Chunks"}
                  </span>
                </div>
              </div>
            </section>

            {/* 4. TIPS PANEL */}
            <section className="bg-zinc-900 border border-zinc-800 rounded-lg p-5 space-y-3">
              <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                Tips
              </h3>
              <ul className="space-y-2 text-xs text-zinc-400">
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400 shrink-0 font-bold">✓</span>
                  <span>Select Hindi, English, Telugu, or Auto Detect before recording.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400 shrink-0 font-bold">✓</span>
                  <span>Answers match your question's language automatically.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400 shrink-0 font-bold">✓</span>
                  <span>Answers are strictly grounded in trusted knowledge.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400 shrink-0 font-bold">✓</span>
                  <span>Conversational phrases are safely answered without triggering false RAG.</span>
                </li>
              </ul>
            </section>

            {/* RESET BUTTON */}
            {result && (
              <button
                onClick={resetDemo}
                className="w-full bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 hover:text-white py-2.5 rounded text-xs font-medium transition flex items-center justify-center gap-1.5 cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Ask another question</span>
              </button>
            )}

          </div>

        </div>

        {/* FOOTER */}
        <footer className="pt-6 border-t border-zinc-800 text-center sm:text-left text-xs text-zinc-500 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Voice RAG • Multilingual Production Knowledge System</span>
          <span className="font-mono text-[11px]">Sarvam STT (hi-IN / en-IN / te-IN) • FAISS/BM25 • Groq llama-3.1-8b-instant</span>
        </footer>

      </div>
    </div>
  )
}
