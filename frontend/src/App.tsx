import React, { useState, useEffect, useRef } from 'react'
import { 
  Mic, Square, Loader2, CheckCircle2, AlertCircle, 
  Search, FileText, Clock, Database, RefreshCw, ShieldCheck, Sparkles, Compass, Volume2
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
    return code || 'Auto Detected'
  }

  const sampleQueries = [
    { text: "कॉर्पोरेशन क्या है?", lang: "hi-IN", label: "Hindi: Corporation" },
    { text: "What is a corporation?", lang: "en-IN", label: "English: Corporation" },
    { text: "కార్పొరేషన్ అంటే ఏమిటి?", lang: "te-IN", label: "Telugu: Corporation" }
  ]

  return (
    <div className="min-h-screen bg-[#FFFDF5] text-[#0B3C2D] flex flex-col justify-between selection:bg-[#FF2A75] selection:text-white font-sans">
      
      {/* BACKGROUND COASTAL DECORATIVE MOTIF (SUBTLE) */}
      <div className="fixed top-0 left-0 right-0 h-2 bg-gradient-to-r from-[#0B3C2D] via-[#FF2A75] to-[#FFC700] z-50" />

      {/* MAIN CONTAINER */}
      <div className="max-w-[1320px] w-full mx-auto px-4 sm:px-6 lg:px-8 pt-8 pb-12 space-y-10 flex-1">
        
        {/* ================================================== */}
        {/* 1. HEADER SECTION */}
        {/* ================================================== */}
        <header className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-6 border-b border-[#0B3C2D]/15 gap-4">
          <div className="flex items-center gap-3">
            {/* HH Goa Identity Badge */}
            <div className="bg-[#0B3C2D] text-[#FFF8E8] px-3.5 py-1.5 rounded-lg font-editorial font-extrabold text-sm tracking-wider flex items-center gap-2 shadow-sm">
              <span className="w-2.5 h-2.5 rounded-full bg-[#FF2A75]" />
              <span>HH GOA 2026</span>
            </div>
            <div>
              <h1 className="text-xl font-extrabold tracking-tight text-[#0B3C2D] font-editorial flex items-center gap-2">
                Voice RAG
              </h1>
              <p className="text-xs text-[#0B3C2D]/70 font-medium">
                Multilingual Knowledge Engine
              </p>
            </div>
          </div>

          {/* System Status Pill */}
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-[#FFF8E8] border border-[#0B3C2D]/20 text-[#0B3C2D] text-xs font-semibold shadow-xs">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#FF2A75] opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-[#FF2A75]"></span>
              </span>
              <span>System Ready</span>
            </span>
          </div>
        </header>

        {/* ================================================== */}
        {/* 2. GOA HERO SECTION */}
        {/* ================================================== */}
        <section className="relative bg-[#FFF8E8] border border-[#E3D9C3] rounded-2xl p-6 sm:p-10 space-y-4 overflow-hidden shadow-xs">
          
          {/* Minimal Horizon / Wave Line SVG Accent */}
          <div className="absolute top-0 right-0 w-96 h-96 opacity-10 pointer-events-none transform translate-x-20 -translate-y-20">
            <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
              <circle cx="100" cy="100" r="80" fill="#FFC700" />
              <path d="M 0 100 Q 50 80, 100 100 T 200 100 L 200 200 L 0 200 Z" fill="#0B3C2D" />
            </svg>
          </div>

          <div className="max-w-3xl space-y-3 relative z-10">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-[#FF2A75]/10 text-[#FF2A75] text-xs font-bold uppercase tracking-wider">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Official HH Goa 2026 Demonstration</span>
            </div>
            
            <h2 className="text-3xl sm:text-5xl font-extrabold text-[#0B3C2D] font-editorial tracking-tight leading-[1.15]">
              Multilingual Intelligence for <span className="underline decoration-[#FF2A75] decoration-4 underline-offset-4">HH Goa 2026</span>
            </h2>
            
            <p className="text-base sm:text-lg text-[#0B3C2D]/80 font-normal leading-relaxed max-w-2xl">
              Ask questions in <strong className="text-[#0B3C2D] font-semibold">English, Hindi, or Telugu</strong>. Speak naturally and retrieve grounded knowledge instantly.
            </p>
          </div>
        </section>

        {/* ================================================== */}
        {/* MAIN TWO-COLUMN RESPONSIVE LAYOUT */}
        {/* ================================================== */}
        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,2.1fr)_minmax(320px,0.9fr)] gap-8 items-start">
          
          {/* ================================================== */}
          {/* LEFT COLUMN: INTERACTION & MAIN CONTENT */}
          {/* ================================================== */}
          <div className="space-y-8">
            
            {/* 3. QUESTION INTERFACE */}
            <section className="bg-[#FFF8E8] border border-[#E3D9C3] rounded-2xl p-6 sm:p-8 space-y-6 shadow-xs">
              
              {/* Header & Language Selector */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-[#0B3C2D]/15">
                <div>
                  <h3 className="text-xl font-bold font-editorial text-[#0B3C2D] flex items-center gap-2">
                    <Compass className="w-5 h-5 text-[#FF2A75]" />
                    <span>Ask a Question</span>
                  </h3>
                  <p className="text-xs text-[#0B3C2D]/70 mt-0.5">
                    Speak via microphone or type your question in English, Hindi, or Telugu.
                  </p>
                </div>

                {/* Language Selector */}
                <div className="flex items-center gap-2 bg-[#FFFDF5] border border-[#0B3C2D]/20 rounded-lg px-3 py-1.5 text-xs font-semibold">
                  <span className="text-[#0B3C2D]/60 uppercase tracking-wider text-[10px] font-bold">Language:</span>
                  <select
                    value={selectedLang}
                    onChange={(e) => setSelectedLang(e.target.value)}
                    aria-label="Select target language"
                    className="bg-transparent text-[#0B3C2D] font-semibold focus:outline-none cursor-pointer text-xs"
                  >
                    {languageOptions.map((opt) => (
                      <option key={opt.code} value={opt.code} className="bg-[#FFFDF5] text-[#0B3C2D]">
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* PROMINENT MICROPHONE HERO INTERACTION CARD */}
              <div className="bg-[#FFFDF5] border border-[#E3D9C3] rounded-xl p-6 text-center flex flex-col items-center justify-center space-y-4">
                
                {uiState === 'RECORDING' ? (
                  <div className="space-y-3">
                    {/* Animated Pulse Recording Button */}
                    <button
                      type="button"
                      onClick={stopRecording}
                      aria-label="Stop recording speech"
                      className="w-24 h-24 rounded-full bg-[#FF2A75] text-white flex items-center justify-center animate-goa-pulse cursor-pointer shadow-lg mx-auto"
                    >
                      <Square className="w-8 h-8 fill-current" />
                    </button>

                    <div className="space-y-1">
                      <p className="text-sm font-bold text-[#FF2A75] tracking-wide uppercase flex items-center justify-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-[#FF2A75] animate-ping" />
                        Listening Active ({formatTime(recordingTime)})
                      </p>
                      <p className="text-xs text-[#0B3C2D]/70 font-medium">
                        Speak clearly in {selectedLang === 'unknown' ? 'your preferred language' : getLanguageDisplayLabel(selectedLang)} — click to finish.
                      </p>
                    </div>
                  </div>
                ) : uiState === 'PROCESSING' ? (
                  <div className="py-4 space-y-3">
                    <div className="w-16 h-16 rounded-full bg-[#FFF8E8] border border-[#0B3C2D]/20 text-[#0B3C2D] flex items-center justify-center mx-auto">
                      <Loader2 className="w-8 h-8 animate-spin text-[#FF2A75]" />
                    </div>
                    <p className="text-sm font-bold text-[#0B3C2D]">
                      Retrieving grounded knowledge...
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <button
                      type="button"
                      onClick={startRecording}
                      aria-label="Microphone input - speak in your language"
                      className="w-24 h-24 rounded-full bg-[#0B3C2D] hover:bg-[#082E22] text-[#FFF8E8] flex items-center justify-center transition-all transform hover:scale-105 cursor-pointer shadow-md mx-auto group border-4 border-[#FFC700]"
                      title="Click to speak"
                    >
                      <Mic className="w-10 h-10 text-[#FF2A75] group-hover:scale-110 transition-transform" />
                    </button>

                    <div className="space-y-1">
                      <p className="text-sm font-bold text-[#0B3C2D] font-editorial">
                        Click Microphone to Speak
                      </p>
                      <p className="text-xs text-[#0B3C2D]/70">
                        Supports natural voice speech in Hindi, English & Telugu
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* TEXT INPUT FORM */}
              <form onSubmit={handleTextSubmit} className="space-y-2">
                <div className="relative flex items-center w-full bg-[#FFFDF5] border border-[#0B3C2D]/20 rounded-xl focus-within:border-[#FF2A75] focus-within:ring-2 focus-within:ring-[#FF2A75]/20 transition-all">
                  <input
                    type="text"
                    value={textQuery}
                    onChange={(e) => setTextQuery(e.target.value)}
                    disabled={uiState === 'RECORDING' || uiState === 'PROCESSING'}
                    placeholder="Or type your question in Hindi (देवनागरी), English, or Telugu (తెలుగు)..."
                    className="w-full bg-transparent px-4 py-3.5 text-sm sm:text-base text-[#0B3C2D] placeholder-[#0B3C2D]/40 focus:outline-none disabled:opacity-50 multilingual-text flex-1"
                  />

                  <div className="pr-2 shrink-0">
                    <button
                      type="submit"
                      disabled={uiState === 'PROCESSING' || uiState === 'RECORDING' || !textQuery.trim()}
                      className="px-5 py-2.5 bg-[#FF2A75] hover:bg-[#FF087F] text-white font-bold text-xs rounded-lg transition disabled:opacity-40 shrink-0 cursor-pointer shadow-xs uppercase tracking-wider"
                    >
                      Search
                    </button>
                  </div>
                </div>
              </form>

              {/* Sample Queries */}
              {uiState !== 'RECORDING' && uiState !== 'PROCESSING' && (
                <div className="pt-3 border-t border-[#0B3C2D]/10 text-xs space-y-2">
                  <span className="font-bold text-[#0B3C2D]/70 uppercase tracking-wider text-[11px]">Sample Queries:</span>
                  <div className="flex flex-wrap gap-2">
                    {sampleQueries.map((sample, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleSampleClick(sample.text, sample.lang)}
                        className="px-3 py-1.5 bg-[#FFFDF5] hover:bg-[#FFF8E8] border border-[#0B3C2D]/20 hover:border-[#FF2A75] rounded-lg text-[#0B3C2D] font-medium transition cursor-pointer text-xs multilingual-text shadow-2xs"
                      >
                        <span className="font-bold text-[#FF2A75] mr-1">{sample.label.split(':')[0]}:</span>
                        "{sample.text}"
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </section>

            {/* ERROR NOTICE */}
            {errorMsg && (
              <div className="p-5 rounded-xl bg-[#FF2A75]/10 border border-[#FF2A75]/30 text-[#0B3C2D] flex items-start gap-3 text-sm">
                <AlertCircle className="w-5 h-5 text-[#FF2A75] shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-bold text-[#FF2A75]">System Notice</h4>
                  <p className="text-xs text-[#0B3C2D]/80 mt-1 font-medium">{errorMsg}</p>
                </div>
              </div>
            )}

            {/* 4. EDITORIAL ANSWER SECTION */}
            {result && (
              <section className="bg-[#FFF8E8] border-l-8 border-l-[#FF2A75] border border-[#E3D9C3] rounded-2xl p-6 sm:p-8 space-y-6 shadow-sm">
                
                {/* Header Metadata */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#0B3C2D]/15 pb-4 gap-2">
                  <div className="flex items-center gap-2">
                    <span className="px-2.5 py-1 bg-[#0B3C2D] text-[#FFC700] text-xs font-bold uppercase tracking-wider rounded">
                      ANSWER
                    </span>
                    <span className="text-xs font-semibold text-[#0B3C2D]/70">
                      Language: {getLanguageDisplayLabel(result.language_code)}
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    {result.grounded ? (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#0B3C2D] text-[#FFFDF5] text-xs font-bold">
                        <CheckCircle2 className="w-3.5 h-3.5 text-[#FFC700]" /> Grounded in Knowledge Base
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#FF2A75]/15 text-[#FF2A75] text-xs font-bold border border-[#FF2A75]/30">
                        <ShieldCheck className="w-3.5 h-3.5" /> Unverified
                      </span>
                    )}
                    <span className="text-xs font-bold text-[#0B3C2D] bg-[#FFFDF5] px-2.5 py-1 rounded-full border border-[#0B3C2D]/15">
                      Confidence: {(result.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                {/* Main Answer Content */}
                <div className="space-y-3">
                  <p className="text-xl sm:text-2xl text-[#0B3C2D] leading-relaxed font-normal multilingual-text">
                    {result.answer}
                  </p>
                </div>

                {/* RECOGNIZED SPEECH TRANSCRIPT */}
                {result.transcript && (
                  <div className="p-4 bg-[#FFFDF5] border border-[#0B3C2D]/15 rounded-xl space-y-1.5">
                    <div className="flex items-center justify-between text-xs font-bold text-[#0B3C2D]/60 uppercase tracking-wider">
                      <span className="flex items-center gap-1.5">
                        <Volume2 className="w-3.5 h-3.5 text-[#FF2A75]" />
                        Recognized Speech
                      </span>
                      <span>Input Language: {getLanguageDisplayLabel(result.language_code)}</span>
                    </div>
                    <p className="text-base font-semibold text-[#0B3C2D] multilingual-text">
                      "{result.transcript}"
                    </p>
                  </div>
                )}
              </section>
            )}

            {/* 5. CANONICAL SOURCES EVIDENCE */}
            {result && (
              <section className="bg-[#FFFDF5] border border-[#E3D9C3] rounded-2xl p-6 sm:p-8 space-y-4 shadow-xs">
                <div className="flex items-center justify-between border-b border-[#0B3C2D]/15 pb-3">
                  <div>
                    <h4 className="text-sm font-bold font-editorial text-[#0B3C2D] uppercase tracking-wider flex items-center gap-2">
                      <FileText className="w-4 h-4 text-[#FF2A75]" />
                      <span>Canonical Evidence Sources ({result.sources?.length || 0})</span>
                    </h4>
                    <p className="text-xs text-[#0B3C2D]/60">
                      MSMARCO multilingual dataset knowledge base verification
                    </p>
                  </div>
                </div>

                {result.sources && result.sources.length > 0 ? (
                  <div className="space-y-3">
                    {result.sources.map((source, idx) => (
                      <div key={idx} className="p-4 bg-[#FFF8E8] border border-[#E3D9C3] rounded-xl space-y-2 text-xs">
                        <div className="flex items-center justify-between font-mono font-semibold text-[#0B3C2D]">
                          <span className="text-[#FF2A75] font-bold">
                            #{String(idx + 1).padStart(2, '0')} &nbsp; Source Chunk • {source.chunk_id}
                          </span>
                          <div className="flex items-center gap-2">
                            <span className="bg-[#FFFDF5] px-2 py-0.5 rounded border border-[#0B3C2D]/15 text-[11px]">
                              Lang: {source.language === 'en' ? 'English' : source.language || 'English'}
                            </span>
                            <span className="bg-[#0B3C2D] text-[#FFF8E8] px-2 py-0.5 rounded text-[11px]">
                              Score: {source.score}
                            </span>
                          </div>
                        </div>
                        <p className="text-xs sm:text-sm text-[#0B3C2D]/90 leading-relaxed multilingual-text pt-1">
                          {source.text}
                        </p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-4 bg-[#FFF8E8] border border-[#E3D9C3] rounded-xl text-xs text-[#0B3C2D]/70 font-medium">
                    No verified sources retrieved for this query.
                  </div>
                )}
              </section>
            )}

          </div>

          {/* ================================================== */}
          {/* RIGHT COLUMN: SIDEBAR METRICS & DETAILS */}
          {/* ================================================== */}
          <div className="space-y-6">

            {/* 6. COMPACT PERFORMANCE METRICS PANEL */}
            {result && (
              <section className="bg-[#FFF8E8] border border-[#E3D9C3] rounded-2xl p-6 space-y-4 shadow-xs">
                <div className="flex items-center justify-between border-b border-[#0B3C2D]/15 pb-2.5">
                  <h3 className="text-xs font-bold text-[#0B3C2D] uppercase tracking-wider flex items-center gap-1.5">
                    <Clock className="w-4 h-4 text-[#FF2A75]" />
                    <span>Performance Metrics</span>
                  </h3>
                  <span className="text-[10px] font-mono font-bold bg-[#FFC700] text-[#0B3C2D] px-2 py-0.5 rounded">
                    Real-time
                  </span>
                </div>

                <div className="space-y-2.5 text-xs font-mono">
                  {result.latency.stt_ms > 0 && (
                    <div className="flex items-center justify-between text-[#0B3C2D]">
                      <span className="font-sans font-medium text-[#0B3C2D]/70">Sarvam STT</span>
                      <span className="font-bold">{result.latency.stt_ms} ms</span>
                    </div>
                  )}
                  <div className="flex items-center justify-between text-[#0B3C2D]">
                    <span className="font-sans font-medium text-[#0B3C2D]/70">Retrieval</span>
                    <span className="font-bold">{result.latency.retrieval_ms} ms</span>
                  </div>
                  <div className="flex items-center justify-between text-[#0B3C2D]">
                    <span className="font-sans font-medium text-[#0B3C2D]/70">Groq LLM</span>
                    <span className="font-bold">{result.latency.llm_ms} ms</span>
                  </div>
                  <div className="flex items-center justify-between text-[#0B3C2D] pt-2 border-t border-[#0B3C2D]/15">
                    <span className="font-sans font-bold">Total Latency</span>
                    <span className="font-extrabold text-[#FF2A75] text-sm">{result.latency.total_ms} ms</span>
                  </div>
                </div>
              </section>
            )}

            {/* TECHNICAL SPECS */}
            <section className="bg-[#FFF8E8] border border-[#E3D9C3] rounded-2xl p-6 space-y-3 shadow-xs">
              <h3 className="text-xs font-bold text-[#0B3C2D] uppercase tracking-wider flex items-center gap-1.5">
                <Database className="w-4 h-4 text-[#0B3C2D]" />
                <span>Architecture Details</span>
              </h3>
              <div className="space-y-2 text-xs text-[#0B3C2D]/80">
                <div className="flex items-center justify-between border-b border-[#0B3C2D]/10 pb-1.5">
                  <span className="text-[#0B3C2D]/60 font-medium">Vector Index</span>
                  <span className="font-mono font-bold text-[#0B3C2D]">FAISS</span>
                </div>
                <div className="flex items-center justify-between border-b border-[#0B3C2D]/10 pb-1.5">
                  <span className="text-[#0B3C2D]/60 font-medium">Sparse Search</span>
                  <span className="font-mono font-bold text-[#0B3C2D]">Rank-BM25</span>
                </div>
                <div className="flex items-center justify-between border-b border-[#0B3C2D]/10 pb-1.5">
                  <span className="text-[#0B3C2D]/60 font-medium">LLM Engine</span>
                  <span className="font-mono font-bold text-[#0B3C2D]">Groq GPT-OSS</span>
                </div>
                <div className="flex items-center justify-between border-b border-[#0B3C2D]/10 pb-1.5">
                  <span className="text-[#0B3C2D]/60 font-medium">Speech-to-Text</span>
                  <span className="font-mono font-bold text-[#0B3C2D]">Sarvam saaras:v3</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[#0B3C2D]/60 font-medium">Languages</span>
                  <span className="font-mono font-bold text-[#FF2A75]">hi-IN, en-IN, te-IN</span>
                </div>
              </div>
            </section>

            {/* RESET ACTION */}
            {result && (
              <button
                onClick={resetDemo}
                className="w-full bg-[#0B3C2D] hover:bg-[#082E22] text-[#FFF8E8] font-bold py-3 rounded-xl text-xs transition cursor-pointer flex items-center justify-center gap-2 shadow-xs uppercase tracking-wider"
              >
                <RefreshCw className="w-4 h-4 text-[#FFC700]" />
                <span>Ask Another Question</span>
              </button>
            )}

          </div>

        </div>

        {/* ================================================== */}
        {/* 7. GOA FOOTER & VISUAL ENDING */}
        {/* ================================================== */}
        <footer className="pt-8 border-t border-[#0B3C2D]/15 space-y-4">
          
          {/* Subtle Wave Line SVG Graphic */}
          <div className="w-full overflow-hidden leading-none opacity-20">
            <svg viewBox="0 0 1200 40" preserveAspectRatio="none" className="w-full h-4">
              <path d="M0,0 C150,30 350,-10 500,10 C650,30 900,-10 1200,0 L1200,40 L0,40 Z" fill="#0B3C2D" />
            </svg>
          </div>

          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-[#0B3C2D]/70 font-medium text-center sm:text-left">
            <div className="flex items-center gap-2">
              <span className="font-bold text-[#0B3C2D]">HH Goa 2026</span>
              <span>•</span>
              <span>Multilingual Voice RAG Demonstration</span>
            </div>
            <div className="font-mono text-[11px]">
              Sarvam STT (hi-IN / en-IN / te-IN) • FAISS / Rank-BM25 • Groq LLM
            </div>
          </div>
        </footer>

      </div>
    </div>
  )
}

