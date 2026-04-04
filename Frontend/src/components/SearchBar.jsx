import { useState } from 'react';
import { Search, Image } from 'lucide-react';

export default function SearchBar({ onSearch, onImageUpload, isLoading }) {
  const [query, setQuery] = useState('');
  const [file, setFile] = useState(null);

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
      // clear input value if desired by resetting the input element via key (handled by parent if needed)
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto my-8 relative z-10">
      <form onSubmit={handleSubmit} className="relative flex items-center">
        <label htmlFor="concept-query" className="sr-only">
          Search concept
        </label>
        <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
          <Search className={`h-5 w-5 ${isLoading ? 'text-primary-500 animate-pulse' : 'text-slate-400'}`} />
        </div>
        <input
          id="concept-query"
          name="conceptQuery"
          type="text"
          className="block w-full pl-12 pr-4 py-4 bg-slate-800/80 backdrop-blur-md border border-slate-700 rounded-2xl leading-5 text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 sm:text-lg transition-all shadow-xl"
          placeholder="What do you want to explore? e.g. 'Human Heart'"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={!query.trim() || isLoading}
          className="absolute right-2 top-2 bottom-2 px-6 bg-primary-500 hover:bg-primary-600 text-white rounded-xl font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Generating...' : 'Generate 3D'}
        </button>
      </form>

      <div className="mt-3 flex items-center gap-3">
        <label htmlFor="image-upload-input" className="flex items-center gap-2 cursor-pointer">
          <Image className="w-5 h-5 text-slate-300" />
          <span className="text-sm text-slate-400">Upload an image to generate 3D</span>
        </label>
        <input id="image-upload-input" name="imageUpload" type="file" accept="image/*" onChange={handleFileChange} />
        <button
          type="button"
          onClick={handleUpload}
          disabled={!file || isLoading}
          className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Processing...' : 'Generate from Image'}
        </button>
      </div>
    </div>
  );
}
