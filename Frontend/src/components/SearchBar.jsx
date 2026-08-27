import { useState, useRef } from 'react';
import { Search, Upload, RotateCcw, Loader2 } from 'lucide-react';

export default function SearchBar({ onSearch, onImageUpload, onReset, isLoading }) {
  const [query, setQuery] = useState('');
  const [file, setFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSearch(query);
    }
  };

  const handleFileChange = (e) => {
    const f = e.target.files && e.target.files[0];
    setFile(f || null);
  };

  const handleUpload = async () => {
    if (file && !isLoading) {
      onImageUpload(file);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleResetClick = () => {
    setQuery('');
    setFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (onReset) onReset();
  };

  return (
    <div className="w-full max-w-2xl mx-auto my-8 relative z-10">
      <form onSubmit={handleSubmit} className="relative flex items-center gap-2">
        <div className="relative flex-1">
          <label htmlFor="concept-query" className="sr-only">
            Search concept
          </label>
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            {isLoading ? (
              <Loader2 className="h-5 w-5 text-[#111111]/60 animate-spin" />
            ) : (
              <Search className="h-5 w-5 text-[#111111]/40" />
            )}
          </div>
          <input
            id="concept-query"
            name="conceptQuery"
            type="text"
            className="block w-full pl-12 pr-[7.5rem] sm:pr-36 py-4 bg-white border border-[#111111]/20 rounded-2xl leading-5 text-[#111111] placeholder-[#111111]/40 placeholder:truncate focus:outline-none focus:ring-2 focus:ring-[#111111]/30 focus:border-[#111111] text-sm sm:text-lg transition-all shadow-sm"
            placeholder="What do you want to explore? e.g. 'Human Heart'"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!query.trim() || isLoading}
            className="absolute right-2 top-2 bottom-2 px-3 sm:px-6 text-xs sm:text-sm bg-[#111111] hover:bg-[#2a2a2a] text-white rounded-xl font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap z-10 flex items-center gap-1.5"
          >
            {isLoading && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
            <span>{isLoading ? 'Generating...' : 'Generate 3D'}</span>
          </button>
        </div>

        <button
          type="button"
          onClick={handleResetClick}
          title="Reset"
          aria-label="Reset"
          disabled={isLoading}
          className="shrink-0 w-12 h-12 sm:h-14 flex items-center justify-center rounded-2xl border border-[#111111]/20 bg-white hover:bg-black/[0.04] disabled:opacity-40 disabled:cursor-not-allowed transition-colors shadow-sm"
        >
          <RotateCcw className="w-4 h-4 sm:w-5 sm:h-5 text-[#111111]/70" />
        </button>
      </form>

      <div className="mt-4 pl-1 flex flex-wrap items-center gap-3">
        <label
          htmlFor="image-upload-input"
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white border border-[#111111]/20 text-[#111111] text-sm font-semibold shadow-sm cursor-pointer hover:bg-black/[0.04] hover:border-[#111111]/40 transition-colors shrink-0"
        >
          <Upload className="w-4 h-4 text-[#111111]/80" />
          <span>Upload Image</span>
        </label>
        <label
          htmlFor="image-upload-input"
          className="flex-1 min-w-[160px] max-w-xs truncate px-3 py-2.5 rounded-xl bg-black/[0.03] border border-[#111111]/15 text-sm text-[#111111]/60 cursor-pointer hover:bg-black/[0.06] hover:border-[#111111]/30 transition-colors"
        >
          {file ? file.name : 'No file selected'}
        </label>
        <input
          ref={fileInputRef}
          id="image-upload-input"
          name="imageUpload"
          type="file"
          accept="image/*"
          className="sr-only"
          onChange={handleFileChange}
        />
        <button
          type="button"
          onClick={handleUpload}
          disabled={!file || isLoading}
          className="px-5 py-2.5 bg-[#111111] hover:bg-[#2a2a2a] text-white text-sm font-medium rounded-xl shadow-sm disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0 flex items-center gap-1.5"
        >
          {isLoading && file && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
          <span>Generate from Image</span>
        </button>
      </div>
    </div>
  );
}
