import { useEffect, useMemo, useState, useRef } from 'react';
import SearchBar from './components/SearchBar';
import ThreeCanvas from './components/ThreeCanvas';
import ModelSelection from './components/ModelSelection';
import ModelDetailModal from './components/ModelDetailModal';
import AIChatbot from './components/AIChatbot';
import ReviewPanel from './components/ReviewPanel';
import { generateNarration, truncateNarrationToTime, calculateNarrationDuration } from './utils/narrationGenerator';
import {
  Volume2,
  Square,
  Sparkles,
  Play,
  ArrowRight,
  Eye,
  Rotate3d,
  MessageSquareText,
  Search,
  Tags,
} from 'lucide-react';
import { apiUrl } from './config/api';
import './App.css';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [modelData, setModelData] = useState(null);
  const [explodedValue, setExplodedValue] = useState(0);
  const [error, setError] = useState(null);
  const [resultsList, setResultsList] = useState([]);
  const [activeQuery, setActiveQuery] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [detailedModel, setDetailedModel] = useState(null);
  const [rightPanelTab, setRightPanelTab] = useState('reviews');

  const studioRef = useRef(null);

  useEffect(() => {
    document.title = 'Concept-2-3D | AI Spatial Studio';
    return () => {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  const narrationText = useMemo(() => {
    if (!modelData) return '';
    const topic = activeQuery || modelData.title || 'this concept';
    let narration = generateNarration(topic, modelData, activeQuery);
    const duration = calculateNarrationDuration(narration);
    if (duration > 120) {
      narration = truncateNarrationToTime(narration, 90, 120);
    }
    return narration;
  }, [modelData, activeQuery]);

  const narrationDuration = useMemo(() => {
    return calculateNarrationDuration(narrationText);
  }, [narrationText]);

  const formatDuration = (seconds) => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleReviewSubmitted = async () => {
    if (resultsList.length === 0) return;
    try {
      const reviewSummaries = await Promise.all(
        resultsList.map(async (model) => {
          try {
            const res = await fetch(apiUrl(`/reviews/${encodeURIComponent(model.uid)}/summary`));
            if (res.ok) {
              const summaryData = await res.json();
              return { uid: model.uid, summary: summaryData.data };
            }
          } catch (err) {
            console.error(`Failed to fetch reviews for ${model.uid}:`, err);
          }
          return { uid: model.uid, summary: null };
        })
      );

      const updatedResults = resultsList.map((model) => {
        const reviewData = reviewSummaries.find((r) => r.uid === model.uid);
        return {
          ...model,
          review_summary: reviewData?.summary || null,
          average_rating: reviewData?.summary?.avg_rating || 0,
        };
      }).sort((a, b) => (b.average_rating || 0) - (a.average_rating || 0));

      setResultsList(updatedResults);
      if (modelData) {
        const updatedCurrentModel = updatedResults.find((m) => m.uid === modelData.uid);
        if (updatedCurrentModel) {
          setModelData(updatedCurrentModel);
        }
      }
    } catch (err) {
      console.error('Failed to re-sort models after review:', err);
    }
  };

  const stopNarration = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    setIsSpeaking(false);
  };

  const playEnglishNarration = () => {
    if (!narrationText || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(narrationText);
    utterance.lang = 'en-US';
    utterance.rate = 1;
    utterance.pitch = 1;

    const voices = window.speechSynthesis.getVoices();
    const englishVoice = voices.find((v) => v.lang && v.lang.toLowerCase().startsWith('en'));
    if (englishVoice) utterance.voice = englishVoice;

    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    setIsSpeaking(true);
    window.speechSynthesis.speak(utterance);
  };

  const scrollToStudio = () => {
    if (studioRef.current) {
      studioRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleSearch = async (query) => {
    const cleanedQuery = (query || '').trim();
    setIsLoading(true);
    setError(null);
    setExplodedValue(0);
    setResultsList([]);
    setActiveQuery(cleanedQuery);
    stopNarration();
    scrollToStudio();

    try {
      const searchRes = await fetch(apiUrl('/search'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: cleanedQuery }),
      });

      const searchJson = await searchRes.json();

      if (searchJson.status === 'fallback') {
        const fallbackData = searchJson.data || {};
        const fallbackModel = {
          uid: 'fallback-procedural',
          title: fallbackData.name || cleanedQuery,
          source: 'procedural',
          is_fallback: true,
          components: fallbackData.components || ['sphere', 'box'],
          part_definitions: fallbackData.part_definitions || [],
          hierarchy: fallbackData.hierarchy || [],
          spatial_breakdown: fallbackData.spatial_breakdown || null,
          category_priors: fallbackData.category_priors || null,
          ai_overview: fallbackData.ai_overview || '',
          score: 85,
        };
        setResultsList([fallbackModel]);
        setModelData(fallbackModel);
      } else if (searchJson.status === 'success') {
        const rawResults = Array.isArray(searchJson.data) ? searchJson.data : [searchJson.data];

        const enrichedResults = await Promise.all(
          rawResults.map(async (model) => {
            try {
              const res = await fetch(apiUrl(`/reviews/${encodeURIComponent(model.uid)}/summary`));
              if (res.ok) {
                const summaryData = await res.json();
                return {
                  ...model,
                  review_summary: summaryData.data,
                  average_rating: summaryData.data?.avg_rating || 0,
                };
              }
            } catch (err) {
              console.error(`Failed to fetch reviews for ${model.uid}:`, err);
            }
            return {
              ...model,
              review_summary: null,
              average_rating: 0,
            };
          })
        );

        enrichedResults.sort((a, b) => (b.average_rating || 0) - (a.average_rating || 0));

        setResultsList(enrichedResults);
        if (enrichedResults.length > 0) {
          setModelData(enrichedResults[0]);
        }
      } else {
        throw new Error(searchJson.message || 'Unknown error');
      }
    } catch (err) {
      console.error(err);
      setError('Failed to contact the backend. Ensure the API server is running and reachable.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageUpload = async (file) => {
    setIsLoading(true);
    setError(null);
    scrollToStudio();
    try {
      const form = new FormData();
      form.append('file', file);

      const res = await fetch(apiUrl('/generate_from_image_async'), {
        method: 'POST',
        body: form,
      });

      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || json.message || 'Failed to start generation');

      const jobId = json.job_id;

      let attempts = 0;
      let jobResp = null;
      while (attempts < 120) {
        await new Promise((r) => setTimeout(r, 2000));
        attempts += 1;
        try {
          const s = await fetch(apiUrl(`/generate_status/${jobId}`));
          jobResp = await s.json();
          if (jobResp.status === 'done') {
            const model = {
              uid: jobResp.uid,
              title: file.name,
              model_url: jobResp.model_url,
              source: 'generated',
            };
            setResultsList([model]);
            setModelData(model);
            break;
          }
          if (jobResp.status === 'failed') {
            throw new Error(jobResp.error || 'Generation failed');
          }
        } catch (err) {
          console.error('Polling error', err);
        }
      }
      if (!jobResp || jobResp.status !== 'done') {
        throw new Error('Generation timed out or failed');
      }
    } catch (err) {
      console.error('Image generation error', err);
      setError(err.message || 'Failed to generate model from image');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setModelData(null);
    setResultsList([]);
    setActiveQuery('');
    setError(null);
    stopNarration();
  };

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#e9e9e7] text-[#111111] selection:bg-[#111111] selection:text-[#e9e9e7]">
      {/* Editorial Typographic Background Wall */}
      <div className="page-wall" aria-hidden="true">
        <span className="page-wall-word" style={{ top: '3%', left: '0%', fontSize: '9vw' }}>
          CONCEPT
        </span>
        <span className="page-wall-word" style={{ top: '18%', left: '68%', fontSize: '7vw' }}>
          IDEA
        </span>
        <span className="page-wall-word" style={{ top: '44%', right: '-1%', fontSize: '8vw' }}>
          FORM
        </span>
        <span className="page-wall-word" style={{ top: '58%', left: '1%', fontSize: '8vw' }}>
          DEPTH
        </span>
        <span className="page-wall-word" style={{ top: '84%', left: '6%', fontSize: '6.5vw' }}>
          SPATIAL
        </span>
        <span className="page-wall-word" style={{ top: '94%', left: '66%', fontSize: '7.5vw' }}>
          REAL
        </span>
      </div>

      {/* Hero Editorial Card Section */}
      <section className="w-full text-[#111111] relative z-10 flex justify-center">
        <div className="w-full max-w-[1400px] bg-[#e9e9e7] border border-[#111111]/20 rounded-2xl overflow-hidden my-3 mx-3 md:my-6 md:mx-6 min-h-[92vh] flex flex-col relative shadow-sm">
          {/* Top Brand Bar */}
          <div className="flex items-center justify-between px-6 md:px-10 py-6">
            <div className="flex items-center gap-2 text-xs md:text-sm font-bold tracking-[0.25em] uppercase">
              <span className="inline-block w-2.5 h-2.5 rotate-45 border-2 border-[#111111]"></span>
              <span>
                CONCEPT-2-3D
                <span className="typing-cursor inline-block w-[2px] h-[0.9em] bg-[#111111] ml-0.5 align-middle" aria-hidden="true"></span>
              </span>
            </div>
            <div className="flex items-center gap-3">
              <span className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#111111]/5 border border-[#111111]/15 text-[11px] font-semibold tracking-wider uppercase text-[#111111]/70">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                Studio Ready
              </span>
              <button
                type="button"
                onClick={scrollToStudio}
                className="px-4 py-1.5 rounded-full bg-[#111111] text-[#e9e9e7] text-xs font-semibold hover:bg-[#2a2a2a] transition-colors"
              >
                Launch Studio
              </button>
            </div>
          </div>

          {/* Hero Main Content */}
          <div className="flex-1 relative px-6 md:px-10 pb-8 flex flex-col justify-between">
            <div>
              <p className="text-xs md:text-sm font-mono tracking-widest text-[#111111]/50 mt-2 mb-4 relative z-10">
                01/05
              </p>
              <h1 className="text-4xl sm:text-5xl md:text-6xl font-black tracking-tighter leading-[0.95] uppercase max-w-xl relative z-10">
                Changing<br />Your Idea Of<br />What 3D Can<br />Do
              </h1>
            </div>

            <div className="relative z-10 mt-6 space-y-6 max-w-lg">
              <p className="text-base sm:text-lg font-bold leading-tight tracking-tight text-[#111111]/85">
                From a single prompt to something you can hold, rotate, and study.
              </p>
              <p className="text-sm text-[#111111]/70 leading-relaxed">
                Every idea has a shape waiting to be found. We give it form, texture, and a stage to be seen from every angle.
              </p>
              <div>
                <button
                  type="button"
                  onClick={scrollToStudio}
                  className="inline-flex items-center gap-3 rounded-full bg-[#111111] text-[#e9e9e7] pl-2 pr-6 py-2.5 hover:bg-[#2a2a2a] transition-all shadow-md hover:scale-[1.02]"
                >
                  <span className="w-8 h-8 rounded-full bg-[#e9e9e7] text-[#111111] flex items-center justify-center shrink-0">
                    <Play className="w-3.5 h-3.5 fill-current" />
                  </span>
                  <span className="text-sm font-semibold tracking-wide">Start generating</span>
                </button>
              </div>
            </div>

            {/* Bottom 3-column Highlights */}
            <div className="grid grid-cols-1 sm:grid-cols-3 border-t border-[#111111]/15 mt-10 pt-6">
              <div className="py-3 sm:pr-6 border-b sm:border-b-0 sm:border-r border-[#111111]/15">
                <p className="text-xs font-semibold tracking-[0.25em] uppercase text-[#111111]/45 mb-1.5">Precision</p>
                <p className="text-xs md:text-sm text-[#111111]/75 leading-relaxed">Every angle considered, nothing left to chance.</p>
              </div>
              <div className="py-3 sm:px-6 border-b sm:border-b-0 sm:border-r border-[#111111]/15">
                <p className="text-xs font-semibold tracking-[0.25em] uppercase text-[#111111]/45 mb-1.5">Speed</p>
                <p className="text-xs md:text-sm text-[#111111]/75 leading-relaxed">Ideas made real before the thought fades.</p>
              </div>
              <div className="py-3 sm:pl-6">
                <p className="text-xs font-semibold tracking-[0.25em] uppercase text-[#111111]/45 mb-1.5">Depth</p>
                <p className="text-xs md:text-sm text-[#111111]/75 leading-relaxed">More than a flat render — something you can walk around.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main Studio Area */}
      <div ref={studioRef} className="relative z-10 min-h-screen flex flex-col items-center overflow-x-hidden text-[#111111] p-4">
        <div className="w-full max-w-[1400px] flex items-center gap-3 mb-4 z-10 pt-4">
          <span className="text-xs font-bold tracking-[0.25em] uppercase text-[#111111]/60">Studio</span>
          <span className="flex-1 h-px bg-[#111111]/15"></span>
        </div>

        <main className="w-full max-w-[1440px] flex-grow flex flex-col z-10 relative">
          <div className="relative overflow-hidden">
            {/* Tagline */}
            <div className={`relative z-10 text-center transition-all duration-500 ${modelData ? 'mb-4 py-2' : 'mb-6 py-8'}`}>
              <h2 className="text-4xl md:text-6xl font-black mb-3 tracking-tighter leading-tight text-[#111111]">
                Think it. <span className="text-[#111111]/60">See it.</span> Explore it.
              </h2>
              {!modelData && (
                <p className="text-[#111111]/65 text-base md:text-lg max-w-2xl mx-auto leading-relaxed">
                  An intelligent pipeline that bridges abstract text and spatial reality.<br />
                  Enter a concept or upload a photo to generate an interactive 3D model.
                </p>
              )}
            </div>

            {/* Search and Upload Controls */}
            <div className="relative z-10 w-full max-w-2xl mx-auto mb-6">
              <SearchBar
                onSearch={handleSearch}
                onImageUpload={handleImageUpload}
                onReset={handleReset}
                isLoading={isLoading}
              />
            </div>

            {error && (
              <div className="w-full max-w-2xl mx-auto mb-6 p-4 bg-rose-50 border border-rose-300 rounded-2xl text-rose-800 text-center text-sm font-medium shadow-sm">
                {error}
              </div>
            )}

            {/* 3D Interactive Spatial Workspace (Active when modelData exists) */}
            {modelData && (
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 w-full h-[calc(100vh-280px)] min-h-[580px] animate-in fade-in zoom-in-95 duration-500 my-4">
                {/* Left Column: Model Selection List */}
                <div className="lg:col-span-1 h-full overflow-hidden bg-white/70 backdrop-blur-md border border-[#111111]/15 rounded-3xl p-3 shadow-md flex flex-col">
                  <ModelSelection
                    results={resultsList}
                    currentModel={modelData}
                    onSelect={(m) => {
                      stopNarration();
                      setModelData(m);
                    }}
                    onShowDetails={(m) => {
                      setDetailedModel(m);
                    }}
                  />
                </div>

                {/* Center Column: 3D Interactive Three.js Canvas */}
                <div className="lg:col-span-2 h-full bg-[#111111] border border-[#111111]/20 rounded-3xl overflow-hidden relative group shadow-xl flex flex-col">
                  <ThreeCanvas modelData={modelData} explodedValue={explodedValue} />

                  {/* Top Floating Model Header */}
                  <div className="absolute top-4 left-4 p-3 bg-black/70 backdrop-blur-md rounded-2xl border border-white/10 text-white flex items-center gap-3 shadow-lg">
                    <div>
                      <h4 className="text-sm font-bold">{modelData.title || 'Interactive 3D Model'}</h4>
                      <p className="text-[11px] text-white/60">Source: {modelData.source || 'Generated'}</p>
                    </div>
                  </div>

                  {/* Bottom Explosion Slider Overlay */}
                  <div className="absolute bottom-4 left-4 right-4 bg-black/70 backdrop-blur-md rounded-2xl border border-white/10 p-3 flex items-center justify-between text-white shadow-lg">
                    <div className="flex items-center gap-3 flex-1 max-w-sm">
                      <span className="text-xs font-semibold tracking-wider uppercase text-white/70 shrink-0">Explode View</span>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={explodedValue}
                        onChange={(e) => setExplodedValue(parseFloat(e.target.value))}
                        className="w-full accent-white cursor-pointer"
                      />
                      <span className="text-xs font-mono text-white/90 shrink-0">{Math.round(explodedValue * 100)}%</span>
                    </div>
                  </div>
                </div>

                {/* Right Column: Audio Narration, Reviews & AI Chatbot */}
                <div className="lg:col-span-1 h-full overflow-hidden flex flex-col gap-4">
                  {/* English Narration Player */}
                  <div className="bg-white/80 backdrop-blur-md border border-[#111111]/15 rounded-3xl p-4 shadow-md">
                    <div className="flex items-center justify-between mb-3 text-[#111111] font-bold">
                      <div className="flex items-center gap-2">
                        <Volume2 size={18} className="text-[#111111]" />
                        <h3 className="text-xs uppercase tracking-wider">Audio Narration</h3>
                      </div>
                      {narrationText && (
                        <span className="text-[11px] font-mono text-[#111111]/70 bg-black/5 px-2 py-0.5 rounded-full">
                          {formatDuration(narrationDuration)}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={playEnglishNarration}
                        disabled={!narrationText}
                        className="flex-1 inline-flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-[#111111] hover:bg-[#2a2a2a] disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-semibold transition-colors shadow-sm"
                      >
                        <Volume2 size={14} />
                        {isSpeaking ? 'Replay' : 'Listen'}
                      </button>
                      <button
                        type="button"
                        onClick={stopNarration}
                        disabled={!isSpeaking}
                        className="inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl bg-black/5 hover:bg-black/10 disabled:opacity-50 disabled:cursor-not-allowed text-[#111111] text-xs font-semibold transition-colors"
                      >
                        <Square size={13} />
                        Stop
                      </button>
                    </div>
                  </div>

                  {/* Tabs: Reviews / AI Chat */}
                  <div className="flex-1 min-h-0 bg-white/80 backdrop-blur-md border border-[#111111]/15 rounded-3xl shadow-md flex flex-col overflow-hidden">
                    <div className="flex items-center border-b border-[#111111]/10 bg-black/[0.02] p-1.5 gap-1">
                      <button
                        onClick={() => setRightPanelTab('reviews')}
                        className={`flex-1 py-2 text-xs font-bold uppercase tracking-wider rounded-xl transition-all ${
                          rightPanelTab === 'reviews'
                            ? 'bg-[#111111] text-white shadow-sm'
                            : 'text-[#111111]/60 hover:text-[#111111]'
                        }`}
                      >
                        Reviews
                      </button>
                      <button
                        onClick={() => setRightPanelTab('chat')}
                        className={`flex-1 py-2 text-xs font-bold uppercase tracking-wider rounded-xl transition-all ${
                          rightPanelTab === 'chat'
                            ? 'bg-[#111111] text-white shadow-sm'
                            : 'text-[#111111]/60 hover:text-[#111111]'
                        }`}
                      >
                        AI Chat
                      </button>
                    </div>

                    <div className="flex-1 min-h-0 overflow-hidden relative p-3">
                      {rightPanelTab === 'reviews' ? (
                        <ReviewPanel
                          modelId={modelData?.uid}
                          onReviewSubmitted={handleReviewSubmitted}
                        />
                      ) : (
                        <AIChatbot
                          modelContext={activeQuery || modelData?.title || modelData?.uid || '3D model viewing'}
                        />
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>

        {/* Feature Showcase: Discover the Vision */}
        <section className="w-full max-w-[1440px] mx-auto px-2 mt-16 mb-8">
          <div className="rounded-[28px] overflow-hidden bg-[#111111] relative shadow-2xl w-full text-white p-8 md:p-14">
            <div className="flex flex-col lg:flex-row justify-between items-start gap-10">
              <div className="max-w-xl">
                <span className="inline-block px-3 py-1 rounded-full bg-white/10 text-white/80 text-[10px] font-mono tracking-widest uppercase mb-4">
                  Discover The Vision
                </span>
                <h3 className="text-4xl sm:text-5xl font-black tracking-tight leading-tight mb-6">
                  See it before you build it.
                </h3>
                <p className="text-white/70 text-sm sm:text-base leading-relaxed mb-6">
                  We believe the gap between imagining something and holding it in your hands should be a sentence, not a skill. Concept-2-3D exists to close that gap — type a concept or drop in a photo, and watch it become a real, explorable 3D object in seconds.
                </p>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2 text-white/80 text-xs font-semibold">
                    <Eye size={16} />
                    <span>Live 3D Rendering</span>
                  </div>
                  <div className="flex items-center gap-2 text-white/80 text-xs font-semibold">
                    <Rotate3d size={16} />
                    <span>Spatial Inspection</span>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full lg:max-w-md">
                <div className="bg-white/5 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
                  <MessageSquareText className="w-6 h-6 text-white/80 mb-3" />
                  <h4 className="text-sm font-bold mb-1">One Prompt, Endless Shapes</h4>
                  <p className="text-xs text-white/60 leading-relaxed">Transforms natural language intent into verified 3D meshes.</p>
                </div>
                <div className="bg-white/5 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
                  <Tags className="w-6 h-6 text-white/80 mb-3" />
                  <h4 className="text-sm font-bold mb-1">Every Part Named & Known</h4>
                  <p className="text-xs text-white/60 leading-relaxed">Automated structural part labeling and spatial breakdown.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 4-Step Interactive Workflow Grid */}
        <section className="w-full max-w-[1440px] mx-auto px-2 my-10">
          <div className="rounded-[28px] overflow-hidden bg-white/70 border border-[#111111]/15 p-8 md:p-12 shadow-sm">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-10 gap-4">
              <div>
                <span className="inline-block bg-black/5 border border-[#111111]/10 rounded-full px-3 py-1 text-xs font-medium text-[#111111]/70 mb-3">
                  Four Steps, No Learning Curve
                </span>
                <h3 className="text-3xl sm:text-4xl font-black tracking-tighter text-[#111111]">
                  How An Idea Becomes Real
                </h3>
              </div>
              <button
                type="button"
                onClick={scrollToStudio}
                className="inline-flex items-center gap-2 bg-[#111111] text-white rounded-full px-5 py-2.5 text-xs font-semibold hover:bg-[#2a2a2a] transition-colors"
              >
                <Sparkles size={14} />
                <span>Start Creating</span>
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Step 1 */}
              <div className="bg-white border border-[#111111]/10 rounded-2xl p-5 shadow-sm flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-9 h-9 rounded-xl bg-[#111111] text-white flex items-center justify-center">
                      <MessageSquareText size={16} />
                    </div>
                    <span className="text-xs font-mono font-bold text-[#111111]/40">01</span>
                  </div>
                  <h4 className="text-sm font-bold text-[#111111] mb-1.5">Describe</h4>
                  <p className="text-xs text-[#111111]/65 leading-relaxed">
                    Type a concept in plain words, or upload a photo of something you already have in mind.
                  </p>
                </div>
              </div>

              {/* Step 2 */}
              <div className="bg-white border border-[#111111]/10 rounded-2xl p-5 shadow-sm flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-9 h-9 rounded-xl bg-[#111111] text-white flex items-center justify-center">
                      <Search size={16} />
                    </div>
                    <span className="text-xs font-mono font-bold text-[#111111]/40">02</span>
                  </div>
                  <h4 className="text-sm font-bold text-[#111111] mb-1.5">Match & Build</h4>
                  <p className="text-xs text-[#111111]/65 leading-relaxed">
                    Your description is searched against a library of real models, or shaped fresh if nothing fits.
                  </p>
                </div>
              </div>

              {/* Step 3 */}
              <div className="bg-white border border-[#111111]/10 rounded-2xl p-5 shadow-sm flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-9 h-9 rounded-xl bg-[#111111] text-white flex items-center justify-center">
                      <Eye size={16} />
                    </div>
                    <span className="text-xs font-mono font-bold text-[#111111]/40">03</span>
                  </div>
                  <h4 className="text-sm font-bold text-[#111111] mb-1.5">Explore in 3D</h4>
                  <p className="text-xs text-[#111111]/65 leading-relaxed">
                    Rotate, zoom, explode, and inspect the result from any angle, rendered live in your browser.
                  </p>
                </div>
              </div>

              {/* Step 4 */}
              <div className="bg-white border border-[#111111]/10 rounded-2xl p-5 shadow-sm flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-9 h-9 rounded-xl bg-[#111111] text-white flex items-center justify-center">
                      <Tags size={16} />
                    </div>
                    <span className="text-xs font-mono font-bold text-[#111111]/40">04</span>
                  </div>
                  <h4 className="text-sm font-bold text-[#111111] mb-1.5">Understand Each Part</h4>
                  <p className="text-xs text-[#111111]/65 leading-relaxed">
                    Every structural piece is labeled and explained, turning a shape into knowledge.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Editorial Footer */}
        <footer className="w-full max-w-[1440px] mt-8 pt-6 pb-8 border-t border-[#111111]/10 text-center">
          <p className="text-xs text-[#111111]/60 leading-relaxed max-w-3xl mx-auto">
            © 2026 Concept-3D • Designed, Developed & Maintained by GuruMachanica. All rights reserved.
          </p>
        </footer>
      </div>

      {/* Model Detail Modal */}
      <ModelDetailModal model={detailedModel} onClose={() => setDetailedModel(null)} />
    </div>
  );
}

export default App;
