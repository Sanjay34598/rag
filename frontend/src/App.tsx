import React, { useState, useEffect, useRef } from 'react'
import { 
  Mic, Square, Loader2, CheckCircle2, AlertCircle, 
  Search, FileText, Clock, Database, RefreshCw, ShieldCheck, 
  Sparkles, Compass, Volume2, BookOpen, Activity
} from 'lucide-react'
import { getApiEndpoint } from './config/api'

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
    { code: 'en-IN', label: 'English (en-IN)' },
    { code: 'hi-IN', label: 'Hindi (hi-IN)' },
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
      const voiceEndpoint = getApiEndpoint('/api/v1/voice/query')
      const res = await fetch(voiceEndpoint, {
        method: 'POST',
        body: formData,
      })

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
      const ragEndpoint = getApiEndpoint('/api/v1/rag/query')
      const res = await fetch(ragEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqBody),
      })

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
    return code || 'Auto Detect'
  }

  const sampleQueries = [
    { text: "What is a corporation?", lang: "en-IN" },
    { text: "कॉर्पोरेशन क्या है?", lang: "hi-IN" },
    { text: "కార్పొరేషన్ అంటే ఏమిటి?", lang: "te-IN" }
  ]

  return (
    <div className="min-h-screen bg-[#FFFDF5] text-[#063B2E] flex flex-col justify-between selection:bg-[#FF2A75] selection:text-white font-sans">
      
      {/* ================================================== */}
      {/* 1. HEADER (DARK GREEN #063B2E) */}
      {/* ================================================== */}
      <header className="bg-[#063B2E] text-[#FFF8E8] px-4 sm:px-8 py-3.5 flex flex-col md:flex-row items-center justify-between gap-4 border-b border-[#074C3A] shadow-md relative z-30">
        <div className="flex items-center gap-4">
          {/* Hibiscus Flower Icon + HH GOA 2026 */}
          <div className="flex items-center gap-2.5">
            <span className="text-2xl">🌺</span>
            <div>
              <div className="font-editorial font-extrabold text-lg text-white tracking-wider leading-none">
                HH GOA 2026
              </div>
              <div className="text-[10px] font-bold text-[#FFD400] tracking-widest uppercase mt-0.5">
                BUILD • HACK • INNOVATE
              </div>
            </div>
          </div>

          {/* Vertical Divider */}
          <div className="h-8 w-px bg-white/20 hidden sm:block" />

          {/* Title */}
          <div className="hidden sm:block">
            <div className="font-editorial font-bold text-sm text-white leading-none">
              Voice RAG
            </div>
            <div className="text-xs text-emerald-200/70 font-medium mt-0.5">
              Multilingual Knowledge Engine
            </div>
          </div>
        </div>

        {/* Right Side Navigation + Status */}
        <div className="flex items-center gap-6 text-xs font-semibold">
          <nav className="flex items-center gap-5">
            <a href="#home" className="text-white font-bold relative py-1 border-b-2 border-[#FF2A75]">Home</a>
            <a href="#about" className="text-emerald-100/80 hover:text-white transition">About</a>
            <a href="#works" className="text-emerald-100/80 hover:text-white transition">How it Works</a>
          </nav>

          <div className="flex items-center gap-2 bg-[#074C3A] border border-emerald-600/30 px-3.5 py-1.5 rounded-full shadow-inner">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-emerald-200 text-xs font-medium">System Ready</span>
          </div>
        </div>
      </header>

      {/* ================================================== */}
      {/* 2. GOA HERO SECTION WITH CINEMATIC LANDSCAPE */}
      {/* ================================================== */}
      <section className="relative min-h-[420px] sm:min-h-[480px] bg-cover bg-center bg-no-repeat flex flex-col items-center justify-start text-center px-4 pt-10 pb-28 space-y-4" style={{ backgroundImage: "url('/goa_hero_bg.jpg')" }}>
        
        {/* Soft golden sunset & tropical background gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-b from-[#063B2E]/70 via-black/25 to-[#FFFDF5] pointer-events-none" />

        <div className="relative z-10 max-w-4xl space-y-3">
          {/* Demonstration Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1 rounded-full bg-[#FF2A75]/20 backdrop-blur-md border border-[#FF2A75]/40 text-[#FF2A75] text-xs font-extrabold uppercase tracking-wider shadow-sm">
            <span>🌴 OFFICIAL HH GOA 2026 DEMONSTRATION</span>
          </div>

          {/* Main Headline */}
          <h1 className="text-4xl sm:text-6xl font-extrabold font-editorial text-white tracking-tight drop-shadow-md leading-[1.1]">
            Multilingual Intelligence <br />
            for <span className="text-[#FF2A75]">HH Goa 2026</span>
          </h1>

          {/* Subtitle */}
          <p className="text-base sm:text-lg text-white/95 font-medium max-w-2xl mx-auto drop-shadow-xs">
            Ask questions in <strong className="text-white font-bold underline decoration-emerald-400 decoration-2">English</strong>, <strong className="text-[#FF2A75] font-bold underline decoration-[#FF2A75] decoration-2">Hindi</strong>, or <strong className="text-white font-bold underline decoration-emerald-400 decoration-2">Telugu</strong>.<br />
            Speak naturally and retrieve grounded knowledge instantly.
          </p>

          {/* Pink Wavy Decoration */}
          <div className="flex justify-center text-[#FF2A75] text-xl font-bold tracking-widest pt-1">
            ~~~~~
          </div>
        </div>
      </section>

      {/* ================================================== */}
      {/* 3. OVERLAPPING SEARCH / VOICE PANEL */}
      {/* ================================================== */}
      <div className="-mt-20 relative z-20 max-w-5xl mx-auto px-4 w-full">
        <div className="bg-[#FFFDF5] border border-[#E8F0DF] rounded-3xl p-6 sm:p-8 shadow-2xl space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-[auto_1fr_auto] gap-6 items-center">
            
            {/* MICROPHONE BUTTON (LEFT) */}
            <div className="flex flex-col items-center justify-center space-y-2">
              <div className="relative flex items-center justify-center">
                {/* Waveform Decoration Ring */}
                <div className="absolute -inset-2 flex items-center justify-between px-1 pointer-events-none opacity-50">
                  <span className="w-1 h-8 bg-[#FF2A75] rounded-full animate-bounce" />
                  <span className="w-1 h-12 bg-[#FFD400] rounded-full animate-pulse" />
                  <span className="w-1 h-10 bg-[#FF2A75] rounded-full animate-bounce" />
                </div>

                {uiState === 'RECORDING' ? (
                  <button
                    type="button"
                    onClick={stopRecording}
                    className="w-20 h-20 rounded-full bg-[#FF2A75] text-white flex items-center justify-center animate-goa-pulse cursor-pointer shadow-lg z-10 border-4 border-[#FFD400]"
                    aria-label="Stop recording"
                  >
                    <Square className="w-8 h-8 fill-current" />
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={startRecording}
                    disabled={uiState === 'PROCESSING'}
                    className="w-20 h-20 rounded-full bg-[#063B2E] hover:bg-[#074C3A] text-white flex items-center justify-center transition transform hover:scale-105 cursor-pointer shadow-lg z-10 border-4 border-[#FFD400] group"
                    title="Click to speak"
                    aria-label="Start recording"
                  >
                    <Mic className="w-9 h-9 text-[#FF2A75] group-hover:scale-110 transition-transform" />
                  </button>
                )}
              </div>

              {/* Timer Display */}
              <span className="text-xs font-mono font-bold text-[#063B2E]">
                {uiState === 'RECORDING' ? formatTime(recordingTime) : '00:00'}
              </span>
            </div>

            {/* INPUT & SAMPLE QUERIES (CENTER) */}
            <div className="space-y-3">
              <form onSubmit={handleTextSubmit} className="relative flex items-center">
                <input
                  type="text"
                  value={textQuery}
                  onChange={(e) => setTextQuery(e.target.value)}
                  disabled={uiState === 'RECORDING' || uiState === 'PROCESSING'}
                  placeholder="Ask anything... (English / हिंदी / తెలుగు)"
                  className="w-full bg-[#FFF8E8] border border-[#DDE8D7] rounded-2xl px-5 py-3.5 pr-12 text-sm sm:text-base text-[#063B2E] placeholder-[#063B2E]/50 focus:outline-none focus:border-[#FF2A75] focus:ring-2 focus:ring-[#FF2A75]/20 font-medium multilingual-text shadow-inner"
                />
                <span className="absolute right-4 text-[#063B2E]/40 text-lg">⌨️</span>
              </form>

              {/* Sample Queries Pills */}
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="text-[#063B2E]/60 font-bold">Try example:</span>
                {sampleQueries.map((sample, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSampleClick(sample.text, sample.lang)}
                    className="px-3 py-1.5 bg-[#FFF8E8] hover:bg-[#E8F0DF] border border-[#DDE8D7] rounded-full text-[#063B2E] font-medium transition cursor-pointer text-xs multilingual-text shadow-2xs flex items-center gap-1.5"
                  >
                    <span>{sample.text}</span>
                    <span className="text-[#FF2A75] font-bold">→</span>
                  </button>
                ))}
              </div>
            </div>

            {/* LANGUAGE & SEARCH BUTTON (RIGHT) */}
            <div className="flex flex-col sm:flex-row lg:flex-col items-stretch lg:items-end gap-3">
              <div className="flex items-center justify-between gap-2 bg-[#FFF8E8] border border-[#DDE8D7] rounded-xl px-3 py-2 text-xs">
                <span className="text-[#063B2E]/60 font-bold uppercase tracking-wider text-[10px]">LANGUAGE:</span>
                <select
                  value={selectedLang}
                  onChange={(e) => setSelectedLang(e.target.value)}
                  className="bg-transparent text-[#063B2E] font-bold focus:outline-none cursor-pointer text-xs"
                >
                  {languageOptions.map((opt) => (
                    <option key={opt.code} value={opt.code} className="bg-[#FFFDF5] text-[#063B2E]">
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              <button
                type="submit"
                onClick={handleTextSubmit}
                disabled={uiState === 'PROCESSING' || uiState === 'RECORDING' || !textQuery.trim()}
                className="px-6 py-3 bg-[#FF2A75] hover:bg-[#FF087F] text-white font-bold text-sm rounded-xl transition cursor-pointer shadow-md flex items-center justify-center gap-2 uppercase tracking-wider disabled:opacity-40"
              >
                <Search className="w-4 h-4" />
                <span>Search</span>
              </button>
            </div>

          </div>
        </div>
      </div>

      {/* ERROR NOTICE */}
      {errorMsg && (
        <div className="max-w-5xl mx-auto px-4 mt-6">
          <div className="p-5 rounded-2xl bg-[#FF2A75]/10 border border-[#FF2A75]/30 text-[#063B2E] flex items-start gap-3 text-sm shadow-xs">
            <AlertCircle className="w-5 h-5 text-[#FF2A75] shrink-0 mt-0.5" />
            <div>
              <h4 className="font-bold text-[#FF2A75]">Processing Notice</h4>
              <p className="text-xs text-[#063B2E]/80 mt-1 font-medium">{errorMsg}</p>
            </div>
          </div>
        </div>
      )}

      {/* ================================================== */}
      {/* 4. MAIN CONTENT AREA (ANSWER + PERFORMANCE + SOURCES) */}
      {/* ================================================== */}
      <main className="max-w-6xl mx-auto px-4 py-12 flex-1 w-full">
        {result ? (
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-8 items-start">
            
            {/* LEFT COLUMN: ANSWER & PERFORMANCE */}
            <div className="space-y-6">
              
              {/* ANSWER CARD */}
              <section className="bg-[#FFFDF5] border border-[#E8F0DF] rounded-2xl p-6 sm:p-8 space-y-6 shadow-sm relative overflow-hidden">
                {/* Subtle Palm Tree Watermark */}
                <div className="absolute right-2 bottom-2 opacity-5 pointer-events-none text-9xl">
                  🌴
                </div>

                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#DDE8D7] pb-4">
                  <div className="flex items-center gap-2">
                    <span className="w-8 h-8 rounded-xl bg-[#063B2E] text-[#FFF8E8] flex items-center justify-center font-bold text-sm font-editorial shadow-xs">
                      A
                    </span>
                    <h3 className="text-xl font-bold font-editorial text-[#063B2E]">Answer</h3>
                  </div>

                  <div className="flex items-center gap-2">
                    {result.grounded ? (
                      <span className="px-3.5 py-1 bg-emerald-100 text-emerald-800 border border-emerald-300 rounded-full text-xs font-bold flex items-center gap-1.5 shadow-2xs">
                        <span className="w-2 h-2 rounded-full bg-emerald-600" />
                        Grounded
                      </span>
                    ) : (
                      <span className="px-3.5 py-1 bg-amber-100 text-amber-800 border border-amber-300 rounded-full text-xs font-bold shadow-2xs">
                        Unverified
                      </span>
                    )}
                    <span className="px-3 py-1 bg-emerald-50 text-emerald-800 border border-emerald-200 rounded-full text-xs font-bold shadow-2xs">
                      Confidence <strong className="text-[#FF2A75]">{(result.confidence * 100).toFixed(0)}%</strong>
                    </span>
                  </div>
                </div>

                <div className="space-y-4">
                  <p className="text-lg sm:text-xl text-[#063B2E] leading-relaxed font-normal multilingual-text">
                    {result.answer}
                  </p>

                  {/* RECOGNIZED SPEECH TRANSCRIPT */}
                  {result.transcript && (
                    <div className="p-4 bg-[#E8F0DF]/60 border border-[#063B2E]/10 rounded-xl space-y-1.5">
                      <div className="flex items-center justify-between text-xs font-bold text-[#063B2E]/70">
                        <span className="flex items-center gap-1.5">
                          <Volume2 className="w-4 h-4 text-[#FF2A75]" />
                          Recognized Speech
                        </span>
                        <span>Detected Language: <strong className="font-mono text-[#063B2E] bg-white/60 px-2 py-0.5 rounded border border-[#063B2E]/10">{getLanguageDisplayLabel(result.language_code)}</strong></span>
                      </div>
                      <p className="text-base font-semibold text-[#063B2E] multilingual-text pt-0.5">
                        "{result.transcript}"
                      </p>
                    </div>
                  )}
                </div>
              </section>

              {/* PERFORMANCE SECTION */}
              <section className="bg-[#FFFDF5] border border-[#E8F0DF] rounded-2xl p-6 space-y-4 shadow-2xs">
                <div className="flex items-center gap-2 border-b border-[#DDE8D7] pb-3">
                  <Clock className="w-5 h-5 text-[#063B2E]" />
                  <h4 className="text-sm font-bold font-editorial text-[#063B2E] uppercase tracking-wider">Performance</h4>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3.5 bg-[#FFF8E8] border border-[#DDE8D7] rounded-xl text-center space-y-1 shadow-2xs">
                    <span className="text-[10px] font-bold text-[#063B2E]/60 uppercase">Retrieval</span>
                    <p className="text-base font-extrabold text-[#063B2E]">{result.latency.retrieval_ms} ms</p>
                  </div>
                  <div className="p-3.5 bg-[#FFF8E8] border border-[#DDE8D7] rounded-xl text-center space-y-1 shadow-2xs">
                    <span className="text-[10px] font-bold text-[#063B2E]/60 uppercase">STT (Sarvam)</span>
                    <p className="text-base font-extrabold text-[#FF2A75]">{result.latency.stt_ms ? `${result.latency.stt_ms} ms` : '— ms'}</p>
                  </div>
                  <div className="p-3.5 bg-[#FFF8E8] border border-[#DDE8D7] rounded-xl text-center space-y-1 shadow-2xs">
                    <span className="text-[10px] font-bold text-[#063B2E]/60 uppercase">LLM (Groq)</span>
                    <p className="text-base font-extrabold text-[#FF2A75]">{result.latency.llm_ms} ms</p>
                  </div>
                  <div className="p-3.5 bg-[#FFF8E8] border border-[#DDE8D7] rounded-xl text-center space-y-1 shadow-2xs">
                    <span className="text-[10px] font-bold text-[#063B2E]/60 uppercase">Total</span>
                    <p className="text-base font-extrabold text-[#063B2E]">{result.latency.total_ms} ms</p>
                  </div>
                </div>
              </section>

              {/* RESET ACTION BUTTON */}
              <button
                onClick={resetDemo}
                className="w-full bg-[#063B2E] hover:bg-[#074C3A] text-[#FFF8E8] font-bold py-3.5 rounded-xl text-xs transition cursor-pointer flex items-center justify-center gap-2 shadow-sm uppercase tracking-wider"
              >
                <RefreshCw className="w-4 h-4 text-[#FFD400]" />
                <span>Ask Another Question</span>
              </button>

            </div>

            {/* RIGHT COLUMN: SOURCES PANEL */}
            <aside className="bg-[#FFFDF5] border border-[#E8F0DF] rounded-2xl p-6 space-y-4 shadow-sm h-fit">
              <div className="flex items-center justify-between border-b border-[#DDE8D7] pb-3">
                <h4 className="text-base font-bold font-editorial text-[#063B2E] flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-[#063B2E]" />
                  <span>Sources ({result.sources?.length || 0})</span>
                </h4>
              </div>

              <div className="space-y-3">
                {result.sources?.map((src, idx) => (
                  <div key={idx} className="p-3.5 bg-[#FFF8E8] border border-[#DDE8D7] rounded-xl space-y-2 text-xs shadow-2xs">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="w-6 h-6 rounded-full bg-[#063B2E] text-white flex items-center justify-center text-[10px] font-bold">
                          {String(idx + 1).padStart(2, '0')}
                        </span>
                        <span className="font-mono font-bold text-[#063B2E] text-xs">{src.chunk_id}</span>
                        <span className="bg-[#063B2E]/10 text-[#063B2E] px-1.5 py-0.5 rounded text-[10px] font-bold">
                          {src.language?.toUpperCase() || 'EN'}
                        </span>
                      </div>
                    </div>
                    <div className="text-[11px] font-bold text-[#FF2A75]">
                      Score: {src.score}
                    </div>
                    <p className="text-xs text-[#063B2E]/80 leading-relaxed multilingual-text">
                      {src.text}
                    </p>
                  </div>
                ))}
              </div>

              <div className="pt-2 border-t border-[#DDE8D7] text-right">
                <button className="text-xs font-bold text-[#FF2A75] hover:underline cursor-pointer">
                  View all sources →
                </button>
              </div>
            </aside>

          </div>
        ) : (
          <div className="text-center py-16 text-[#063B2E]/60 space-y-2">
            <p className="text-base font-medium">
              Ready for your query in English, Hindi, or Telugu.
            </p>
            <p className="text-xs">
              Click the microphone button or type a question above to explore grounded knowledge.
            </p>
          </div>
        )}
      </main>

      {/* ================================================== */}
      {/* 5. FOOTER (DARK GREEN WITH COASTAL SCENE) */}
      {/* ================================================== */}
      <footer className="bg-[#063B2E] text-[#FFF8E8] pt-10 pb-8 px-4 relative overflow-hidden border-t border-[#074C3A]">
        {/* Coastal silhouette backdrop graphic */}
        <div className="max-w-6xl mx-auto flex flex-col items-center justify-center space-y-4 text-center relative z-10">
          
          <div className="flex items-center justify-center gap-3 text-lg font-editorial font-bold text-white">
            <span>🌺</span>
            <span>HH Goa 2026</span>
            <span className="text-[#FF2A75]">|</span>
            <span>Multilingual Voice RAG Demonstration</span>
            <span>🌴</span>
          </div>
          
          <div className="text-xs text-emerald-200/70 font-mono">
            Sarvam STT (hi-IN / en-IN / te-IN) • FAISS / Rank-BM25 • Groq LLM
          </div>
        </div>
      </footer>

    </div>
  )
}


